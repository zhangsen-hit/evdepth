"""
测试DSEC数据加载

验证处理后的DSEC数据是否可以正常加载和训练

https://dsec.ifi.uzh.ch/dsec-datasets/download/

"""
import random
import torch
from pathlib import Path
import yaml
from omegaconf import OmegaConf
import numpy as np

# 导入数据模块
from modules.data.event_data_module import DataModule
from data.utils.types import DataType


def ev_repr_to_img(
    x: np.ndarray,
    use_last_k_bins: int = 2,
    polarity_mode: str = "diff",  # "diff" | "pos" | "neg" | "abs_diff"
    denoise_percentile: float = 90.0,
    color_mode: str = "red_blue",  # "red_blue" | "grayscale"
) -> np.ndarray:
    """
    将 stacked histogram 事件表示转为 RGB 图像 (H, W, 3)，仅用于可视化。

    - 默认红蓝配色：红=正极性事件，蓝=负极性事件（事件相机典型展示方式）；
    - 默认只使用最后 use_last_k_bins 个时间 bin，轮廓更清晰；
    - 支持不同极性组合方式（差分/只看正极/只看负极/绝对差）；
    - 使用百分位归一化 + 简单强度阈值去除低强度噪声点。
    """
    if x.ndim == 3:
        ch, ht, wd = x.shape
    else:
        ch, ht, wd = x.shape[-3:]
    assert ch > 1 and ch % 2 == 0

    half = ch // 2
    num_time_bins = half

    # 选择用于可视化的时间 bin：默认仅使用最后 use_last_k_bins 个
    if use_last_k_bins is None or use_last_k_bins <= 0 or use_last_k_bins >= num_time_bins:
        sel_slice = slice(0, num_time_bins)  # 使用所有 bin
    else:
        sel_slice = slice(num_time_bins - use_last_k_bins, num_time_bins)

    # 正负极性累积
    img_neg = np.asarray(x[:half][sel_slice].sum(axis=0), dtype=np.float32)
    img_pos = np.asarray(x[half:][sel_slice].sum(axis=0), dtype=np.float32)

    if color_mode.lower() == "red_blue":
        # 典型红蓝配色：红=正极性，蓝=负极性
        def _norm(v):
            pl, ph = np.percentile(v, [2, 98])
            if ph > pl:
                return np.clip((v - pl) / (ph - pl), 0, 1)
            return np.zeros_like(v, dtype=np.float32)
        r = _norm(img_pos)
        b = _norm(img_neg)
        g = np.zeros_like(r)
        if denoise_percentile is not None and 0 < denoise_percentile < 100:
            thr_r = np.percentile(r, denoise_percentile)
            thr_b = np.percentile(b, denoise_percentile)
            if thr_r > 0:
                r = np.where(r >= thr_r, r, 0)
            if thr_b > 0:
                b = np.where(b >= thr_b, b, 0)
        img_rgb = np.stack([r * 255, g * 255, b * 255], axis=-1).astype(np.uint8)
        return img_rgb

    polarity_mode = polarity_mode.lower()
    if polarity_mode == "pos":
        img = img_pos
    elif polarity_mode == "neg":
        img = img_neg
    elif polarity_mode == "abs_diff":
        img = np.abs(img_pos - img_neg)
    else:  # "diff"
        img = img_pos - img_neg

    # 百分位归一化，保留对比度
    p_low, p_high = np.percentile(img, [2, 98])
    if p_high > p_low:
        img_norm = np.clip((img - p_low) / (p_high - p_low), 0, 1)
    else:
        img_norm = np.zeros_like(img, dtype=np.float32)
        if np.any(np.isfinite(img)):
            img_norm[img > 0] = 1.0

    gray = (img_norm * 255).astype(np.uint8)

    # 简单去噪：过滤掉强度较低的像素，突出轮廓
    if denoise_percentile is not None and 0 < denoise_percentile < 100:
        thr = np.percentile(gray, denoise_percentile)
        if thr > 0:
            gray = gray.copy()
            gray[gray < thr] = 0

    img_rgb = np.stack([gray, gray, gray], axis=-1)
    return img_rgb


def depth_to_colormap(depth: np.ndarray, mask: np.ndarray = None, vmin: float = 0.5, vmax: float = 80.0) -> np.ndarray:
    """将深度图转为 RGB 伪彩色 (H, W, 3)，无效区域为黑色"""
    import matplotlib.cm as cm
    depth = np.asarray(depth, dtype=np.float32)
    out = np.zeros((*depth.shape, 3), dtype=np.uint8)

    finite = np.isfinite(depth)
    pos = depth > 0

    if mask is not None:
        valid = finite & pos & np.asarray(mask, dtype=bool) & (depth >= vmin) & (depth <= vmax)
    else:
        valid = finite & pos & (depth >= vmin) & (depth <= vmax)

    if vmax <= vmin or not np.any(valid):
        return out

    log_vmin = np.log(vmin)
    log_vmax = np.log(vmax)
    denom = max(log_vmax - log_vmin, 1e-12)

    # log 空间着色：红近蓝远（因此反转归一化）
    log_depth = np.log(np.clip(depth, vmin, vmax))
    norm = np.clip((log_depth - log_vmin) / denom, 0.0, 1.0)
    norm_rev = 1.0 - norm
    colored = (cm.get_cmap("jet")(norm_rev)[:, :, :3] * 255).astype(np.uint8)

    out[valid] = colored[valid]
    return out


def visualize_ev_depth_overlay(
    batch: dict, output_dir: Path, sample_idx: int = 0, frame_idx: int = -1, pair_suffix: str = ""
):
    """
    可视化事件图与深度图叠加，用于直观验证像素级对齐。
    生成三张图：事件图、深度图、叠加图（并排对比）。
    pair_suffix: 用于多对图时的文件名后缀，如 "001"
    """
    data = batch.get('data', batch)
    if DataType.EV_REPR not in data or DataType.DEPTH not in data:
        print("   [SKIP] 无法可视化：batch 中缺少 EV_REPR 或 DEPTH")
        return

    ev_repr = data[DataType.EV_REPR]
    depth = data[DataType.DEPTH]
    mask = data.get(DataType.DEPTH_MASK)

    # 支持序列格式：list of tensors
    if isinstance(ev_repr, (list, tuple)):
        ev_repr = ev_repr[frame_idx]
    if isinstance(depth, (list, tuple)):
        depth = depth[frame_idx]
    if mask is not None and isinstance(mask, (list, tuple)):
        mask = mask[frame_idx]

    # 取单个样本
    if ev_repr.dim() == 4:
        ev_repr = ev_repr[sample_idx]
    if depth.dim() == 4:
        depth = depth[sample_idx]
    if mask is not None and mask.dim() == 4:
        mask = mask[sample_idx]

    ev_np = ev_repr.cpu().numpy()
    depth_np = depth.cpu().numpy()
    if depth_np.ndim == 3:
        depth_np = depth_np[0]
    mask_np = mask.cpu().numpy().squeeze() if mask is not None else None

    # 深度在 log 空间，转回线性
    depth_real = np.exp(depth_np.astype(np.float32))

    ev_img = ev_repr_to_img(ev_np)
    depth_img = depth_to_colormap(depth_real, mask_np)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 叠加图：事件为底，深度半透明叠加上去
    alpha = 0.5
    overlay = (ev_img.astype(np.float32) * (1 - alpha) + depth_img.astype(np.float32) * alpha).astype(np.uint8)

    # 若 depth 有效区域少，在叠加时仅在有 mask 处画深度颜色
    if mask_np is not None and np.any(mask_np):
        overlay = ev_img.copy()
        valid = mask_np > 0
        overlay[valid] = (
            ev_img[valid].astype(np.float32) * (1 - alpha) + depth_img[valid].astype(np.float32) * alpha
        ).astype(np.uint8)

    sfx = f"_{pair_suffix}" if pair_suffix else ""
    try:
        import cv2
        ev_path = output_dir / f"01_events{sfx}.png"
        depth_path = output_dir / f"02_depth{sfx}.png"
        overlay_path = output_dir / f"03_overlay_ev_depth{sfx}.png"
        cv2.imwrite(str(ev_path), cv2.cvtColor(ev_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(depth_path), cv2.cvtColor(depth_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        if not pair_suffix:
            print(f"\n   [OK] 可视化已保存到: {output_dir}")
            print(f"        - 01_events.png: 事件图")
            print(f"        - 02_depth.png: 深度图（伪彩色）")
            print(f"        - 03_overlay_ev_depth.png: 叠加图（边缘对齐则物体轮廓应重合）")
    except ImportError:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(ev_img)
        axes[0].set_title("Events")
        axes[0].axis("off")
        axes[1].imshow(depth_img)
        axes[1].set_title("Depth (colormap)")
        axes[1].axis("off")
        axes[2].imshow(overlay)
        axes[2].set_title("Overlay (edges aligned?)")
        axes[2].axis("off")
        plt.tight_layout()
        save_path = output_dir / f"ev_depth_overlay{sfx}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        if not pair_suffix:
            print(f"\n   [OK] 可视化已保存到: {save_path}")


def check_data_structure(data_path):
    """检查数据结构是否完整"""
    data_path = Path(data_path)
    print("\n" + "=" * 70)
    print("检查数据结构")
    print("=" * 70)
    
    # 检查目录
    for split in ['train', 'val', 'test']:
        split_dir = data_path / split
        if split_dir.exists():
            sequences = list(split_dir.glob('*_seq_*'))
            print(f"\n{split.upper()} split:")
            print(f"  目录: {split_dir}")
            print(f"  序列数: {len(sequences)}")
            
            for seq_dir in sequences:
                print(f"\n  序列: {seq_dir.name}")
                
                # 检查深度数据
                depth_dir = seq_dir / 'depth_v2'
                if depth_dir.exists():
                    depth_files = list(depth_dir.glob('*.h5'))
                    print(f"    深度文件: {len(depth_files)} 个")
                    for f in depth_files:
                        print(f"      - {f.name}")
                else:
                    print(f"    [ERROR] 缺少 depth_v2 目录")
                
                # 检查事件表示
                event_dir = seq_dir / 'event_representations_v2'
                if event_dir.exists():
                    event_subdirs = list(event_dir.glob('stacked_histogram*'))
                    print(f"    事件表示: {len(event_subdirs)} 个")
                    for d in event_subdirs:
                        event_files = list(d.glob('*.h5'))
                        print(f"      - {d.name}: {len(event_files)} 个文件")
                else:
                    print(f"    [ERROR] 缺少 event_representations_v2 目录")


def test_single_batch(config, data_path: str = None):
    """测试加载单个batch"""
    print("\n" + "=" * 70)
    print("测试数据加载")
    print("=" * 70)
    
    # 创建数据模块
    print("\n[1/4] 创建数据模块...")
    data_module = DataModule(
        dataset_config=config.dataset,
        num_workers_train=0,  # Windows下使用0
        num_workers_eval=0,
        batch_size_train=2,  # 小batch测试
        batch_size_eval=2
    )
    print("   [OK] 数据模块创建成功")
    
    # 准备数据
    print("\n[2/4] 准备数据...")
    try:
        data_module.setup(stage='fit')
        print("   [OK] 数据准备完成")
    except Exception as e:
        print(f"   [ERROR] 数据准备失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 获取训练数据加载器
    print("\n[3/4] 获取训练数据加载器...")
    try:
        train_loader = data_module.train_dataloader()
        # Streaming dataloader可能没有len()
        try:
            print(f"   [OK] 训练集大小: {len(train_loader)} batches")
        except TypeError:
            print(f"   [OK] 训练数据加载器创建成功 (streaming模式)")
    except Exception as e:
        print(f"   [ERROR] 获取数据加载器失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 加载一个batch
    print("\n[4/4] 加载测试batch...")
    try:
        batch = next(iter(train_loader))
        print("   [OK] Batch加载成功")
        # stream 模式下 batch = {'data': {...}, 'worker_id': ...}
        data = batch.get('data', batch)

        # 检查数据内容
        print("\n" + "=" * 70)
        print("数据检查")
        print("=" * 70)

        if DataType.EV_REPR in data:
            ev_repr = data[DataType.EV_REPR]
            t = ev_repr[0] if isinstance(ev_repr, (list, tuple)) else ev_repr
            print(f"\n[OK] 事件表示:")
            print(f"   形状: {[x.shape for x in ev_repr[:3]]}..." if isinstance(ev_repr, (list, tuple)) else f"   {ev_repr.shape}")
            print(f"   类型: {t.dtype}")
            print(f"   范围: [{t.min():.2f}, {t.max():.2f}]")
            print(f"   非零元素: {(t != 0).sum().item()} / {t.numel()}")
        
        if DataType.DEPTH in data:
            depth = data[DataType.DEPTH]
            if isinstance(depth, list):
                print(f"\n[OK] 深度图 (序列):")
                print(f"   序列长度: {len(depth)}")
                print(f"   每帧形状: {depth[0].shape}")
                print(f"   类型: {depth[0].dtype}")
                # 统计第一帧
                d = depth[0]
                valid_depth = d[d > 0]
                if len(valid_depth) > 0:
                    print(f"   深度范围: [{valid_depth.min():.2f}, {valid_depth.max():.2f}] 米")
                    print(f"   平均深度: {valid_depth.mean():.2f} 米")
            else:
                print(f"\n[OK] 深度图:")
                print(f"   形状: {depth.shape}")
                print(f"   类型: {depth.dtype}")
                valid_depth = depth[depth > 0]
                if len(valid_depth) > 0:
                    print(f"   深度范围: [{valid_depth.min():.2f}, {valid_depth.max():.2f}] 米")
                    print(f"   平均深度: {valid_depth.mean():.2f} 米")
        
        if DataType.DEPTH_MASK in data:
            mask = data[DataType.DEPTH_MASK]
            if mask is not None:
                if isinstance(mask, list):
                    print(f"\n[OK] 深度掩码 (序列):")
                    print(f"   序列长度: {len(mask)}")
                    print(f"   每帧形状: {mask[0].shape}")
                    m = mask[0]
                    valid_ratio = m.float().mean() * 100
                    print(f"   有效像素比例: {valid_ratio:.1f}%")
                else:
                    print(f"\n[OK] 深度掩码:")
                    print(f"   形状: {mask.shape}")
                    valid_ratio = mask.float().mean() * 100
                    print(f"   有效像素比例: {valid_ratio:.1f}%")

        # 可视化叠加：随机挑选 10 对图验证事件与深度像素级对齐
        print("\n[5/5] 随机挑选 10 对图，生成事件-深度叠加可视化...")
        output_dir = Path(data_path or config.dataset.path) / "alignment_check"
        num_pairs = 10
        max_batches = 50  # 最多遍历的 batch 数，用于随机采样
        batches = [batch]
        loader_iter = iter(train_loader)
        for _ in range(max_batches - 1):
            try:
                b = next(loader_iter)
                batches.append(b)
            except StopIteration:
                break
        sampled = random.sample(batches, min(num_pairs, len(batches)))
        for i, b in enumerate(sampled):
            visualize_ev_depth_overlay(
                b, output_dir=output_dir, sample_idx=0, frame_idx=-1, pair_suffix=f"{i+1:03d}"
            )
        print(f"\n   [OK] 已保存 {len(sampled)} 对叠加图到: {output_dir}")
        print(f"        - 01_events_001.png ~ 01_events_{len(sampled):03d}.png")
        print(f"        - 02_depth_001.png ~ 02_depth_{len(sampled):03d}.png")
        print(f"        - 03_overlay_ev_depth_001.png ~ 03_overlay_ev_depth_{len(sampled):03d}.png")

        print("\n" + "=" * 70)
        print("[OK] 数据加载测试成功!")
        print("=" * 70)
        return True
        
    except Exception as e:
        print(f"   [ERROR] 加载batch失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 70)
    print("DSEC数据加载测试")
    print("=" * 70)
    
    # 加载配置
    config_path = 'config_dsec.yaml'
    print(f"\n加载配置: {config_path}")
    
    if not Path(config_path).exists():
        print(f"[ERROR] 配置文件不存在: {config_path}")
        return
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    config = OmegaConf.create(config_dict)
    
    # 检查数据路径
    data_path = config.dataset.path
    print(f"数据路径: {data_path}")
    
    if not Path(data_path).exists():
        print(f"\n[ERROR] 数据路径不存在: {data_path}")
        print("\n请先运行数据处理脚本:")
        print("  python scripts/process_dsec_data.py \\")
        print("    --input_dir data/DSEC/00 \\")
        print("    --output_dir data/dsec_processed \\")
        print("    --sequence_name train_seq_000")
        return
    
    # 检查数据结构
    check_data_structure(data_path)
    
    # 测试数据加载
    success = test_single_batch(config, data_path=data_path)
    
    if success:
        print("\n" + "=" * 70)
        print("[SUCCESS] 所有测试通过!")
        print("=" * 70)
        print("\n现在可以开始训练:")
        print("  python train.py --config config_dsec.yaml")
    else:
        print("\n" + "=" * 70)
        print("[ERROR] 测试失败，请检查错误信息")
        print("=" * 70)


if __name__ == '__main__':
    main()


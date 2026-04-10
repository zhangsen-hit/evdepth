"""
DSEC数据集预处理脚本 - 仅深度模式
专门用于Windows环境，跳过事件数据处理，仅处理深度数据

这个脚本只处理：
1. 视差图 → 深度图转换
2. 生成深度掩码
3. 创建占位符事件表示（零填充）

适用场景：
- Windows环境无法读取events.h5
- 仅用于测试深度估计模型架构
- 或作为第一步，事件数据稍后在Linux上处理
"""

import argparse
import h5py
import numpy as np
import yaml
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def load_calibration(calib_file):
    """加载相机标定参数"""
    with open(calib_file, 'r') as f:
        calib = yaml.safe_load(f)
    
    # 获取左事件相机的标定参数 (camRect0)
    cam_rect = calib['intrinsics']['camRect0']
    fx = cam_rect['camera_matrix'][0]
    fy = cam_rect['camera_matrix'][1]
    cx = cam_rect['camera_matrix'][2]
    cy = cam_rect['camera_matrix'][3]
    
    # 从disparity_to_depth获取基线和焦距
    Q = np.array(calib['disparity_to_depth']['cams_03'])
    baseline = 1.0 / Q[3, 2]
    f = Q[2, 3]
    
    print(f"相机标定参数:")
    print(f"  焦距: fx={fx:.2f}, fy={fy:.2f}")
    print(f"  主点: cx={cx:.2f}, cy={cy:.2f}")
    print(f"  基线: {baseline:.4f} m")
    print(f"  用于深度计算的焦距: {f:.2f}")
    
    return {'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy, 'baseline': baseline, 'f': f}


def disparity_to_depth(disparity, baseline, focal_length):
    """将视差转换为深度"""
    # 转换为实际视差值
    disparity_float = disparity.astype(np.float32) / 256.0
    
    # 创建有效掩码
    valid_mask = disparity_float > 0.1
    
    # 计算深度
    depth = np.zeros_like(disparity_float)
    depth[valid_mask] = (baseline * focal_length) / disparity_float[valid_mask]
    
    # 限制深度范围
    depth = np.clip(depth, 0.5, 80.0)
    valid_mask = valid_mask & (depth >= 0.5) & (depth <= 80.0)
    
    return depth, valid_mask


def load_disparity_timestamps(timestamp_file):
    """加载视差图时间戳"""
    timestamps = []
    with open(timestamp_file, 'r') as f:
        for line in f:
            timestamps.append(int(line.strip()))
    return np.array(timestamps)


def create_zero_event_representation(height, width, num_bins=10):
    """创建零事件表示（占位符）"""
    return np.zeros((2 * num_bins, height, width), dtype=np.float32)


def process_dsec_depth_only(
    input_dir,
    output_dir,
    sequence_name,
    num_bins=10,
    downsample=True,
):
    """
    仅处理DSEC深度数据（跳过事件）
    
    Args:
        input_dir: DSEC原始数据目录
        output_dir: 输出目录
        sequence_name: 序列名称
        num_bins: 事件表示bins数量（仅用于占位符）
        downsample: 是否下采样
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir) / sequence_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n处理序列: {sequence_name} [仅深度模式]")
    print(f"  输入: {input_path}")
    print(f"  输出: {output_path}")
    print(f"  [WARNING] 事件数据将使用零填充（占位符）")
    
    # 加载标定参数
    calib_file = input_path / 'interlaken_00_c_calibration' / 'cam_to_cam.yaml'
    calib = load_calibration(calib_file)
    
    # 加载视差图时间戳
    timestamp_file = input_path / 'interlaken_00_c_disparity_timestamps.txt'
    disparity_timestamps = load_disparity_timestamps(timestamp_file)
    print(f"  视差图数量: {len(disparity_timestamps)}")
    
    # 原始分辨率
    orig_height, orig_width = 480, 640
    
    # 目标分辨率
    if downsample:
        target_height, target_width = 240, 320
        ds_suffix = '_ds2_nearest'
    else:
        target_height, target_width = orig_height, orig_width
        ds_suffix = ''
    
    # 创建输出目录
    depth_dir = output_path / 'depth_v2'
    depth_dir.mkdir(exist_ok=True)
    
    event_repr_name = f'stacked_histogram_dt=50_nbins={num_bins}'
    event_repr_dir = output_path / 'event_representations_v2' / event_repr_name
    event_repr_dir.mkdir(parents=True, exist_ok=True)
    
    # 准备输出数据
    num_frames = len(disparity_timestamps)
    
    depth_maps = []
    depth_masks = []
    event_reprs = []
    timestamps_us = []
    
    disparity_dir = input_path / 'interlaken_00_c_disparity_event'
    
    print(f"\n生成数据（仅深度）...")
    for idx in tqdm(range(num_frames), desc="处理帧"):
        # 加载视差图
        disparity_file = disparity_dir / f'{idx*2:06d}.png'
        if not disparity_file.exists():
            print(f"警告: 视差图不存在: {disparity_file}")
            continue
        
        disparity_img = Image.open(disparity_file)
        disparity = np.array(disparity_img)
        
        # 转换为深度
        depth, mask = disparity_to_depth(
            disparity,
            calib['baseline'],
            calib['f']
        )
        
        # 下采样
        if downsample:
            depth_pil = Image.fromarray(depth.astype(np.float32))
            mask_pil = Image.fromarray(mask.astype(np.uint8) * 255)
            
            depth_pil = depth_pil.resize((target_width, target_height), Image.NEAREST)
            mask_pil = mask_pil.resize((target_width, target_height), Image.NEAREST)
            
            depth = np.array(depth_pil)
            mask = (np.array(mask_pil) > 128).astype(bool)
        
        depth_maps.append(depth)
        depth_masks.append(mask)
        
        # 时间戳
        current_time_ns = disparity_timestamps[idx]
        timestamps_us.append(current_time_ns // 1000)
        
        # 创建零事件表示（占位符）
        event_repr = create_zero_event_representation(
            target_height, target_width, num_bins
        )
        event_reprs.append(event_repr)
    
    # 保存深度数据
    print(f"\n保存深度数据...")
    depth_maps_array = np.array(depth_maps, dtype=np.float32)
    depth_masks_array = np.array(depth_masks, dtype=bool)
    
    with h5py.File(depth_dir / f'depth_maps{ds_suffix}.h5', 'w') as f:
        f.create_dataset('data', data=depth_maps_array, compression='gzip')
    
    with h5py.File(depth_dir / f'depth_masks{ds_suffix}.h5', 'w') as f:
        f.create_dataset('data', data=depth_masks_array, compression='gzip')
    
    np.save(depth_dir / 'timestamps_us.npy', np.array(timestamps_us))
    
    print(f"  深度图: {depth_maps_array.shape}")
    print(f"  掩码: {depth_masks_array.shape}")
    print(f"  有效深度比例: {depth_masks_array.mean()*100:.1f}%")
    
    # 保存事件表示（零填充）
    print(f"\n保存事件表示（零填充占位符）...")
    event_reprs_array = np.array(event_reprs, dtype=np.float32)
    
    with h5py.File(event_repr_dir / f'event_representations{ds_suffix}.h5', 'w') as f:
        f.create_dataset('data', data=event_reprs_array, compression='gzip')
    
    objframe_idx_2_repr_idx = np.arange(len(event_reprs), dtype=np.int64)
    np.save(event_repr_dir / 'objframe_idx_2_repr_idx.npy', objframe_idx_2_repr_idx)
    np.save(event_repr_dir / 'timestamps_us.npy', np.array(timestamps_us))
    
    print(f"  事件表示: {event_reprs_array.shape} (全零)")
    print(f"  [NOTE] 事件表示为零填充，仅用于测试模型架构")
    
    print(f"\n[OK] 序列 {sequence_name} 处理完成（仅深度模式）!")
    print(f"\n[IMPORTANT] 重要提示:")
    print(f"   - 深度数据: [OK] 真实DSEC深度")
    print(f"   - 事件数据: [WARNING] 零填充（无真实事件）")
    print(f"   - 适用于: 测试模型架构、验证数据加载")
    print(f"   - 不适用于: 实际训练（需要真实事件数据）")


def main():
    parser = argparse.ArgumentParser(description='处理DSEC数据集（仅深度模式）')
    parser.add_argument('--input_dir', type=str, default='data/DSEC/00',
                        help='DSEC原始数据目录')
    parser.add_argument('--output_dir', type=str, default='data/dsec_processed',
                        help='输出目录')
    parser.add_argument('--sequence_name', type=str, default='train_seq_000',
                        help='序列名称')
    parser.add_argument('--num_bins', type=int, default=10,
                        help='事件表示bins数量（占位符）')
    parser.add_argument('--no_downsample', action='store_true',
                        help='不下采样到320x240')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("DSEC数据集预处理 - 仅深度模式")
    print("=" * 70)
    print("\n[WARNING] 此模式跳过事件数据处理（Windows兼容）")
    print("   事件表示将使用零填充")
    print("   生成的数据可用于：")
    print("   - 测试模型架构")
    print("   - 验证数据加载流程")
    print("   - 深度估计模块开发")
    print("\n如需真实事件数据训练，请：")
    print("   - 在Linux环境处理完整数据")
    print("   - 或等待h5py Windows兼容性改进")
    print("=" * 70)
    
    process_dsec_depth_only(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        sequence_name=args.sequence_name,
        num_bins=args.num_bins,
        downsample=not args.no_downsample,
    )
    
    print("\n" + "=" * 70)
    print("[SUCCESS] 处理完成（仅深度模式）!")
    print("=" * 70)
    print("\n下一步:")
    print("  1. 测试数据加载: python test_dsec_loading.py")
    print("  2. 测试模型架构: python train.py --config config_dsec.yaml --max_epochs 1")
    print("\n[WARNING] 提醒: 由于事件数据为零，模型训练效果会受限")
    print("   建议仅用于架构测试和调试")
    print("=" * 70)


if __name__ == '__main__':
    main()


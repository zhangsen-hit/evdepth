"""
测试 LiOSAM odometry2 depth_dataset 数据加载

验证 index.txt + npz 数据是否可以正常加载和训练。
用法: python test_liosam_loading.py
"""
import yaml
from pathlib import Path

from omegaconf import OmegaConf

from modules.data.event_data_module import DataModule
from data.utils.types import DataType, DatasetSamplingMode
from modules.data.merge_mixed_batches import merge_mixed_batches


def test_single_batch(config):
    """测试加载单个 batch"""
    print("\n" + "=" * 70)
    print("测试数据加载")
    print("=" * 70)

    print("\n[1/4] 创建数据模块...")
    data_module = DataModule(
        dataset_config=config.dataset,
        num_workers_train=0,
        num_workers_eval=0,
        batch_size_train=2,
        batch_size_eval=2,
    )
    print("   [OK] 数据模块创建成功")

    print("\n[2/4] 准备数据...")
    try:
        data_module.setup(stage='fit')
        print("   [OK] 数据准备完成")
    except Exception as e:
        print(f"   [ERROR] 数据准备失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n[3/4] 获取训练数据加载器...")
    try:
        train_loader = data_module.train_dataloader()
        if isinstance(train_loader, dict):
            print("   [OK] 训练数据加载器 (MIXED: stream + random)")
        else:
            print("   [OK] 训练数据加载器创建成功")
    except Exception as e:
        print(f"   [ERROR] 获取数据加载器失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n[4/4] 加载测试 batch...")
    try:
        if isinstance(train_loader, dict):
            stream_loader = train_loader[DatasetSamplingMode.STREAM]
            rnd_loader = train_loader[DatasetSamplingMode.RANDOM]
            batch_s = next(iter(stream_loader))
            batch_r = next(iter(rnd_loader))
            batch = merge_mixed_batches({
                DatasetSamplingMode.RANDOM: batch_r,
                DatasetSamplingMode.STREAM: batch_s,
            })
        else:
            batch = next(iter(train_loader))
        data = batch.get('data', batch)
        print("   [OK] Batch 加载成功")

        print("\n" + "=" * 70)
        print("数据检查")
        print("=" * 70)

        if DataType.EV_REPR in data:
            ev_repr = data[DataType.EV_REPR]
            if isinstance(ev_repr, list):
                print(f"\n[OK] 事件表示 (序列):")
                print(f"   序列长度: {len(ev_repr)}")
                print(f"   单帧形状: {ev_repr[0].shape}")
                print(f"   类型: {ev_repr[0].dtype}")
            else:
                print(f"\n[OK] 事件表示: {ev_repr.shape}")

        if DataType.DEPTH in data:
            depth = data[DataType.DEPTH]
            if isinstance(depth, list):
                print(f"\n[OK] 深度图 (序列):")
                print(f"   序列长度: {len(depth)}")
                print(f"   单帧形状: {depth[0].shape}")
            else:
                print(f"\n[OK] 深度图: {depth.shape}")

        if DataType.DEPTH_MASK in data:
            mask = data[DataType.DEPTH_MASK]
            if isinstance(mask, list):
                print(f"\n[OK] 深度掩码 (序列): {len(mask)} 帧")
            else:
                print(f"\n[OK] 深度掩码: {mask.shape}")

        print("\n" + "=" * 70)
        print("[OK] LiOSAM 数据加载测试成功!")
        print("=" * 70)
        return True

    except Exception as e:
        print(f"   [ERROR] 加载 batch 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 70)
    print("LiOSAM odometry2 depth_dataset 数据加载测试")
    print("=" * 70)

    config_path = Path('config_liosam.yaml')
    if not config_path.exists():
        print(f"\n[ERROR] 配置文件不存在: {config_path}")
        return

    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    config = OmegaConf.create(config_dict)

    data_path = Path(config.dataset.path)
    print(f"\n数据路径: {data_path}")

    if not data_path.exists():
        print(f"\n[ERROR] 数据路径不存在: {data_path}")
        return

    train_scenes = config.dataset.get('train_scenes', None)
    val_scenes = config.dataset.get('val_scenes', None)
    if train_scenes is not None and val_scenes is not None:
        all_scenes = list(train_scenes) + list(val_scenes)
        print(f"多场景模式: train={list(train_scenes)}, val={list(val_scenes)}")
        for scene_name in all_scenes:
            scene_path = data_path / str(scene_name)
            if not scene_path.is_dir():
                print(f"\n[ERROR] 场景目录不存在: {scene_path}")
                return
            idx_file = scene_path / 'index.txt'
            if not idx_file.exists():
                print(f"\n[ERROR] 缺少 index.txt: {idx_file}")
                return
        print(f"所有 {len(all_scenes)} 个场景目录及 index.txt 均存在")
    else:
        index_file = data_path / 'index.txt'
        if not index_file.exists():
            print(f"\n[ERROR] 缺少 index.txt: {index_file}")
            return
        print(f"单场景模式, index.txt: 存在")

    success = test_single_batch(config)
    if success:
        print("\n可以开始训练:")
        print("  python train.py --config config_liosam.yaml")
    else:
        print("\n测试失败，请检查错误信息。")


if __name__ == '__main__':
    main()

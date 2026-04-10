"""
生成Virtual（虚拟/合成）事件相机深度数据用于测试

Virtual数据集特点：
- 完全合成的事件表示和深度真值
- 可配置的场景类型和参数
- 用于快速验证代码正确性
- 不需要真实硬件采集

这个脚本生成：
1. 合成的事件表示数据（stacked histogram格式）
2. 对应的深度真值（带有几何规律）
3. 深度有效掩码
4. 必要的元数据文件

生成的数据可以直接用于训练/验证流程的测试
"""
import argparse
import h5py
import numpy as np
from pathlib import Path
from typing import Tuple
import shutil


def create_synthetic_event_representation(
    height: int,
    width: int,
    num_bins: int = 10,
    pattern_type: str = 'moving_gradient'
) -> np.ndarray:
    """
    创建合成的事件表示
    
    Args:
        height: 图像高度
        width: 图像宽度
        num_bins: 时间bin数量
        pattern_type: 模式类型 ('moving_gradient', 'checkerboard', 'random')
    
    Returns:
        event_repr: (num_bins*2, H, W) 事件表示
    """
    channels = num_bins * 2  # positive and negative events
    event_repr = np.zeros((channels, height, width), dtype=np.float32)
    
    if pattern_type == 'moving_gradient':
        # 创建移动的梯度模式（模拟移动物体）
        for c in range(channels):
            x_offset = c * width // channels
            xx, yy = np.meshgrid(
                np.arange(width) - x_offset,
                np.arange(height)
            )
            # 创建径向梯度
            gradient = np.exp(-((xx/width*4)**2 + (yy/height*4)**2))
            event_repr[c] = gradient * np.random.uniform(0.5, 1.0)
    
    elif pattern_type == 'checkerboard':
        # 创建棋盘模式
        block_size = max(height // 8, width // 8)
        for c in range(channels):
            phase = c * np.pi / channels
            for i in range(0, height, block_size):
                for j in range(0, width, block_size):
                    if ((i // block_size + j // block_size) % 2) == 0:
                        event_repr[c, i:i+block_size, j:j+block_size] = \
                            np.sin(phase) * 0.5 + 0.5
    
    elif pattern_type == 'random':
        # 随机噪声
        event_repr = np.random.uniform(0, 1, (channels, height, width)).astype(np.float32)
    
    # 添加一些随机噪声使其更真实
    event_repr += np.random.normal(0, 0.05, event_repr.shape).astype(np.float32)
    event_repr = np.clip(event_repr, 0, 1)
    
    return event_repr


def create_synthetic_depth_map(
    height: int,
    width: int,
    depth_type: str = 'plane_with_objects',
    min_depth: float = 1.0,
    max_depth: float = 50.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    创建合成的深度图和掩码
    
    Args:
        height: 图像高度
        width: 图像宽度
        depth_type: 深度类型
        min_depth: 最小深度（米）
        max_depth: 最大深度（米）
    
    Returns:
        depth_map: (H, W) 深度图（米）
        mask: (H, W) 有效掩码
    """
    depth_map = np.zeros((height, width), dtype=np.float32)
    
    if depth_type == 'plane':
        # 简单的倾斜平面
        yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        depth_map = min_depth + (max_depth - min_depth) * (yy / height * 0.7 + xx / width * 0.3)
    
    elif depth_type == 'plane_with_objects':
        # 背景平面 + 前景物体
        yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        # 背景：倾斜平面
        depth_map = min_depth + (max_depth - min_depth) * (yy / height * 0.5 + 0.3)
        
        # 添加几个矩形物体（更近的深度）
        objects = [
            (height//4, height//2, width//4, width//2, min_depth + 5),  # 左上物体
            (height//2, 3*height//4, 2*width//3, 5*width//6, min_depth + 8),  # 右下物体
            (height//3, 2*height//3, width//3, 2*width//3, min_depth + 3),  # 中心物体
        ]
        
        for y1, y2, x1, x2, obj_depth in objects:
            depth_map[y1:y2, x1:x2] = obj_depth
    
    elif depth_type == 'steps':
        # 阶梯状深度（模拟楼梯）
        num_steps = 5
        step_height = height // num_steps
        for i in range(num_steps):
            depth = min_depth + (max_depth - min_depth) * (i / num_steps)
            y_start = i * step_height
            y_end = (i + 1) * step_height
            depth_map[y_start:y_end, :] = depth
    
    elif depth_type == 'sphere':
        # 球形物体
        yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        center_y, center_x = height // 2, width // 2
        radius = min(height, width) // 3
        
        # 背景平面
        depth_map[:] = max_depth * 0.7
        
        # 球形深度
        dist_sq = (xx - center_x)**2 + (yy - center_y)**2
        sphere_mask = dist_sq <= radius**2
        # 半球深度分布
        depth_map[sphere_mask] = min_depth + 15 - np.sqrt(
            np.maximum(0, radius**2 - dist_sq[sphere_mask])
        ) / radius * 10
    
    # 创建掩码（标记无效区域，如边缘）
    mask = np.ones((height, width), dtype=bool)
    
    # 边缘10%设为无效（模拟视野外）
    border = min(height, width) // 10
    mask[:border, :] = False
    mask[-border:, :] = False
    mask[:, :border] = False
    mask[:, -border:] = False
    
    # 随机添加一些无效区域（模拟反射、遮挡等）
    num_invalid_regions = np.random.randint(2, 5)
    for _ in range(num_invalid_regions):
        y = np.random.randint(border, height - border)
        x = np.random.randint(border, width - border)
        size = np.random.randint(10, 30)
        y1, y2 = max(0, y - size), min(height, y + size)
        x1, x2 = max(0, x - size), min(width, x + size)
        mask[y1:y2, x1:x2] = False
    
    return depth_map, mask


def create_sequence_data(
    output_dir: Path,
    num_frames: int = 100,
    height: int = 240,
    width: int = 320,
    num_bins: int = 10,
    downsample_by_factor_2: bool = True,
    sequence_name: str = 'synthetic_sequence'
):
    """
    创建完整的合成序列数据
    
    Args:
        output_dir: 输出目录
        num_frames: 帧数
        height: 原始高度
        width: 原始宽度
        num_bins: 事件bin数量
        downsample_by_factor_2: 是否降采样
        sequence_name: 序列名称
    """
    # 创建序列目录
    seq_dir = output_dir / sequence_name
    seq_dir.mkdir(parents=True, exist_ok=True)
    
    # 如果降采样，调整尺寸
    if downsample_by_factor_2:
        height_ds = height // 2
        width_ds = width // 2
        ds_suffix = '_ds2_nearest'
    else:
        height_ds = height
        width_ds = width
        ds_suffix = ''
    
    print(f"Creating sequence: {sequence_name}")
    print(f"  Frames: {num_frames}")
    print(f"  Size: {height_ds}x{width_ds}")
    print(f"  Downsample: {downsample_by_factor_2}")
    
    # ==================== 事件表示 ====================
    print("\n[1/4] Generating event representations...")
    ev_repr_dir = seq_dir / 'event_representations_v2' / f'stacked_histogram_dt=50_nbins={num_bins}'
    ev_repr_dir.mkdir(parents=True, exist_ok=True)
    
    event_reprs = []
    depth_types = ['plane_with_objects', 'steps', 'sphere']
    
    for i in range(num_frames):
        # 变换模式使数据多样化
        pattern = 'moving_gradient' if i % 3 == 0 else 'checkerboard'
        ev_repr = create_synthetic_event_representation(
            height_ds, width_ds, num_bins, pattern
        )
        event_reprs.append(ev_repr)
        
        if (i + 1) % 20 == 0:
            print(f"  Generated {i+1}/{num_frames} event representations")
    
    event_reprs = np.stack(event_reprs, axis=0)  # (N, C, H, W)
    
    # 保存事件表示
    ev_repr_file = ev_repr_dir / f'event_representations{ds_suffix}.h5'
    with h5py.File(str(ev_repr_file), 'w') as f:
        f.create_dataset('data', data=event_reprs, compression='gzip', compression_opts=4)
    print(f"  Saved to {ev_repr_file}")
    
    # 创建 objframe_idx_2_repr_idx (所有帧都有对应的表示)
    objframe_idx_2_repr_idx = np.arange(num_frames, dtype=np.int32)
    np.save(str(ev_repr_dir / 'objframe_idx_2_repr_idx.npy'), objframe_idx_2_repr_idx)
    
    # 创建时间戳（假设50ms间隔）
    timestamps_us = np.arange(num_frames) * 50000  # 微秒
    np.save(str(ev_repr_dir / 'timestamps_us.npy'), timestamps_us)
    
    # ==================== 深度数据 ====================
    print("\n[2/4] Generating depth maps...")
    depth_dir = seq_dir / 'depth_v2'
    depth_dir.mkdir(parents=True, exist_ok=True)
    
    depth_maps = []
    depth_masks = []
    
    for i in range(num_frames):
        # 循环使用不同的深度类型
        depth_type = depth_types[i % len(depth_types)]
        
        # 添加时间变化（模拟物体移动或相机运动）
        min_depth = 1.0 + np.sin(i / num_frames * 2 * np.pi) * 0.5
        max_depth = 50.0 + np.cos(i / num_frames * 2 * np.pi) * 10
        
        depth_map, mask = create_synthetic_depth_map(
            height_ds, width_ds, depth_type, min_depth, max_depth
        )
        depth_maps.append(depth_map)
        depth_masks.append(mask)
        
        if (i + 1) % 20 == 0:
            print(f"  Generated {i+1}/{num_frames} depth maps")
    
    depth_maps = np.stack(depth_maps, axis=0)  # (N, H, W)
    depth_masks = np.stack(depth_masks, axis=0)  # (N, H, W)
    
    # 保存深度数据
    depth_file = depth_dir / f'depth_maps{ds_suffix}.h5'
    with h5py.File(str(depth_file), 'w') as f:
        f.create_dataset('data', data=depth_maps, compression='gzip', compression_opts=4)
    print(f"  Saved to {depth_file}")
    
    mask_file = depth_dir / f'depth_masks{ds_suffix}.h5'
    with h5py.File(str(mask_file), 'w') as f:
        f.create_dataset('data', data=depth_masks, compression='gzip', compression_opts=4)
    print(f"  Saved to {mask_file}")
    
    # 深度时间戳（与事件相同）
    np.save(str(depth_dir / 'timestamps_us.npy'), timestamps_us)
    
    # ==================== Labels (深度估计不需要目标检测标签) ====================
    # 注意：深度估计任务不需要labels_v2目录，sequence_base.py会自动处理
    print("\n[3/4] Skipping labels (depth estimation mode)...")
    
    # ==================== 统计信息 ====================
    print("\n[4/4] Computing statistics...")
    print(f"\nSequence created successfully!")
    print(f"  Location: {seq_dir}")
    print(f"  Frames: {num_frames}")
    print(f"  Event representation shape: {event_reprs.shape}")
    print(f"  Depth map shape: {depth_maps.shape}")
    print(f"  Depth range: {depth_maps.min():.2f} - {depth_maps.max():.2f} meters")
    print(f"  Valid pixels: {depth_masks.sum() / depth_masks.size * 100:.1f}%")
    
    return seq_dir


def create_dataset_splits(
    output_dir: Path,
    num_train: int = 3,
    num_val: int = 1,
    num_test: int = 1,
    **kwargs
):
    """
    创建完整的数据集（train/val/test splits）
    """
    print("=" * 60)
    print("Creating Synthetic Depth Dataset")
    print("=" * 60)
    
    # 创建train数据
    train_dir = output_dir / 'train'
    train_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}")
    print("TRAIN SET")
    print('='*60)
    for i in range(num_train):
        create_sequence_data(
            train_dir,
            sequence_name=f'train_seq_{i:03d}',
            **kwargs
        )
    
    # 创建val数据
    val_dir = output_dir / 'val'
    val_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}")
    print("VALIDATION SET")
    print('='*60)
    for i in range(num_val):
        create_sequence_data(
            val_dir,
            sequence_name=f'val_seq_{i:03d}',
            **kwargs
        )
    
    # 创建test数据
    test_dir = output_dir / 'test'
    test_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}")
    print("TEST SET")
    print('='*60)
    for i in range(num_test):
        create_sequence_data(
            test_dir,
            sequence_name=f'test_seq_{i:03d}',
            **kwargs
        )
    
    print("\n" + "=" * 60)
    print("Dataset creation complete!")
    print("=" * 60)
    print(f"\nDataset location: {output_dir}")
    print(f"  Train sequences: {num_train}")
    print(f"  Val sequences: {num_val}")
    print(f"  Test sequences: {num_test}")
    print(f"\nYou can now use this dataset for training:")
    print(f"  python train_depth.py dataset.path={output_dir} ...")


def main():
    parser = argparse.ArgumentParser(
        description='Generate Virtual (synthetic) event camera depth dataset'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/virtual_depth',
        help='Output directory for Virtual dataset'
    )
    parser.add_argument(
        '--num_train',
        type=int,
        default=3,
        help='Number of training sequences'
    )
    parser.add_argument(
        '--num_val',
        type=int,
        default=1,
        help='Number of validation sequences'
    )
    parser.add_argument(
        '--num_test',
        type=int,
        default=1,
        help='Number of test sequences'
    )
    parser.add_argument(
        '--num_frames',
        type=int,
        default=100,
        help='Number of frames per sequence'
    )
    parser.add_argument(
        '--height',
        type=int,
        default=240,
        help='Image height (before downsampling)'
    )
    parser.add_argument(
        '--width',
        type=int,
        default=320,
        help='Image width (before downsampling)'
    )
    parser.add_argument(
        '--num_bins',
        type=int,
        default=10,
        help='Number of time bins for event representation'
    )
    parser.add_argument(
        '--no_downsample',
        action='store_true',
        help='Do not downsample by factor 2'
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    create_dataset_splits(
        output_dir=output_dir,
        num_train=args.num_train,
        num_val=args.num_val,
        num_test=args.num_test,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        num_bins=args.num_bins,
        downsample_by_factor_2=not args.no_downsample,
    )


if __name__ == '__main__':
    main()


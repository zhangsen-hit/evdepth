"""
DSEC数据集预处理脚本
将DSEC格式转换为项目所需的格式：
1. 从events.h5生成事件表示
2. 将视差图转换为深度图
3. 生成masks
4. 组织为项目数据格式

DSEC数据集结构：
- events.h5: 事件数据 (x, y, t, p)
- interlaken_00_d_disparity_event/*.png: 视差图 (uint16)
- interlaken_00_d_disparity_timestamps.txt: 视差图时间戳
- cam_to_cam.yaml: 相机标定参数

生成的数据结构：
- depth_v2/depth_maps_ds2_nearest.h5: 深度图 (米)
- depth_v2/depth_masks_ds2_nearest.h5: 有效深度掩码
- event_representations_v2/*/event_representations_ds2_nearest.h5: 事件表示
"""

import argparse
import h5py
import numpy as np
import yaml
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torch

def load_calibration(calib_file):
    """加载相机标定参数"""
    with open(calib_file, 'r') as f:
        calib = yaml.safe_load(f)
    
    # 获取左事件相机的标定参数 (camRect0)
    cam_rect = calib['intrinsics']['camRect0']
    fx = cam_rect['camera_matrix'][0]  # 焦距x
    fy = cam_rect['camera_matrix'][1]  # 焦距y
    cx = cam_rect['camera_matrix'][2]  # 主点x
    cy = cam_rect['camera_matrix'][3]  # 主点y
    
    # 从disparity_to_depth获取基线和焦距
    # Q矩阵: [[1, 0, 0, -cx], [0, 1, 0, -cy], [0, 0, 0, f], [0, 0, 1/baseline, 0]]
    Q = np.array(calib['disparity_to_depth']['cams_03'])
    baseline = 1.0 / Q[3, 2]  # 基线距离(米)
    f = Q[2, 3]  # 焦距(像素)
    
    print(f"相机标定参数:")
    print(f"  焦距: fx={fx:.2f}, fy={fy:.2f}")
    print(f"  主点: cx={cx:.2f}, cy={cy:.2f}")
    print(f"  基线: {baseline:.4f} m")
    print(f"  用于深度计算的焦距: {f:.2f}")
    
    return {'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy, 'baseline': baseline, 'f': f}


def disparity_to_depth(disparity, baseline, focal_length):
    """
    将视差转换为深度
    
    DSEC视差图格式: uint16, 实际视差 = value / 256.0
    深度公式: depth = baseline * focal_length / disparity
    """
    # 转换为实际视差值
    disparity_float = disparity.astype(np.float32) / 256.0
    
    # 创建有效掩码(视差>0的像素)
    valid_mask = disparity_float > 0.1  # 最小视差阈值
    
    # 计算深度 (米)
    depth = np.zeros_like(disparity_float)
    depth[valid_mask] = (baseline * focal_length) / disparity_float[valid_mask]
    
    # 限制深度范围 (DSEC典型范围: 0.5m - 80m)
    depth = np.clip(depth, 0.5, 80.0)
    
    # 更新掩码(去除超出范围的值)
    valid_mask = valid_mask & (depth >= 0.5) & (depth <= 80.0)
    
    return depth, valid_mask


def load_disparity_timestamps(timestamp_file):
    """加载视差图时间戳"""
    timestamps = []
    with open(timestamp_file, 'r') as f:
        for line in f:
            timestamps.append(int(line.strip()))
    return np.array(timestamps)


def load_events_in_time_window(events_file, start_time_ns, end_time_ns, max_events=1000000):
    """
    加载指定时间窗口内的事件
    
    Args:
        events_file: events.h5文件路径
        start_time_ns: 开始时间(纳秒)
        end_time_ns: 结束时间(纳秒)
        max_events: 最大事件数量(防止内存溢出)
    """
    try:
        # 注意：勿用 driver='core', backing_store=False，否则会将整个 events.h5（可能数 GB）一次性读入内存，导致长时间卡住
        with h5py.File(events_file, 'r') as f:
            # DSEC事件数据结构
            t_offset = f['t_offset'][()]
            ms_to_idx = f['ms_to_idx'][:]
            
            # 转换为毫秒索引
            start_ms = int((start_time_ns - t_offset) // 1000)
            end_ms = int((end_time_ns - t_offset) // 1000)
            
            # 使用ms_to_idx快速定位
            start_ms = max(0, min(start_ms, len(ms_to_idx) - 1))
            end_ms = max(0, min(end_ms, len(ms_to_idx) - 1))
            
            start_idx = ms_to_idx[start_ms]
            end_idx = ms_to_idx[end_ms]
            
            # 确保索引有效
            if start_idx >= end_idx:
                return np.array([], dtype=np.uint16), np.array([], dtype=np.uint16), \
                       np.array([], dtype=np.int64), np.array([], dtype=np.bool_)
            
            num_events = end_idx - start_idx
            # 使用切片读取（start:end）而非高级索引，HDF5 切片读取是连续 I/O，速度快
            # 高级索引 f[arr] 会逐元素访问，在 GB 级数据上极慢导致卡住
            x = np.asarray(f['events/x'][start_idx:end_idx])
            y = np.asarray(f['events/y'][start_idx:end_idx])
            t = np.asarray(f['events/t'][start_idx:end_idx])
            p = np.asarray(f['events/p'][start_idx:end_idx])
            # 若事件过多则均匀采样
            if num_events > max_events:
                step = num_events // max_events
                idx = np.arange(0, num_events, step, dtype=np.int64)[:max_events]
                x, y, t, p = x[idx], y[idx], t[idx], p[idx]
        
        return x, y, t, p
    
    except Exception as e:
        # 如果加载失败，返回空数组
        print(f"    警告: 无法加载事件数据: {e}")
        return np.array([], dtype=np.uint16), np.array([], dtype=np.uint16), \
               np.array([], dtype=np.int64), np.array([], dtype=np.bool_)


def events_to_stacked_histogram(x, y, t, p, height=480, width=640, num_bins=10):
    """
    将事件转换为堆叠直方图表示
    
    Args:
        x, y: 事件坐标
        t: 事件时间戳
        p: 事件极性 (0或1)
        height, width: 图像尺寸
        num_bins: 时间bins数量
    
    Returns:
        histogram: (2*num_bins, height, width) - 正负极性分别累积
    """
    if len(x) == 0:
        return np.zeros((2 * num_bins, height, width), dtype=np.float32)
    
    # 归一化时间到[0, num_bins)
    t_min, t_max = t.min(), t.max()
    if t_max > t_min:
        t_norm = (t - t_min) / (t_max - t_min) * num_bins
        t_norm = np.clip(t_norm, 0, num_bins - 0.001).astype(np.int32)
    else:
        t_norm = np.zeros_like(t, dtype=np.int32)
    
    # 创建直方图
    histogram = np.zeros((2 * num_bins, height, width), dtype=np.float32)
    
    # 分别累积正极性和负极性事件
    for pol in [0, 1]:
        mask = (p == pol)
        if not np.any(mask):
            continue
        
        x_pol = x[mask]
        y_pol = y[mask]
        t_pol = t_norm[mask]
        
        # 确保坐标在范围内
        valid = (x_pol >= 0) & (x_pol < width) & (y_pol >= 0) & (y_pol < height)
        x_pol = x_pol[valid]
        y_pol = y_pol[valid]
        t_pol = t_pol[valid]
        
        # 累积到对应的bin
        for bin_idx in range(num_bins):
            bin_mask = (t_pol == bin_idx)
            if np.any(bin_mask):
                channel_idx = pol * num_bins + bin_idx
                np.add.at(histogram[channel_idx], (y_pol[bin_mask], x_pol[bin_mask]), 1)
    
    return histogram


def process_dsec_sequence(
    input_dir,
    output_dir,
    sequence_name,
    dt_ms=50,  # 事件累积时间窗口(毫秒)
    num_bins=10,
    downsample=True,
):
    """
    处理单个DSEC序列
    
    Args:
        input_dir: DSEC原始数据目录 (包含events.h5等)
        output_dir: 输出目录
        sequence_name: 序列名称 (如 train_seq_000)
        dt_ms: 事件累积时间窗口(毫秒)
        num_bins: 事件表示bins数量
        downsample: 是否下采样到320x240
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir) / sequence_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n处理序列: {sequence_name}")
    print(f"  输入: {input_path}")
    print(f"  输出: {output_path}")
    
    # 加载标定参数
    calib_file = input_path / 'interlaken_00_d_calibration' / 'cam_to_cam.yaml'
    calib = load_calibration(calib_file)
    
    # 加载视差图时间戳
    timestamp_file = input_path / 'interlaken_00_d_disparity_timestamps.txt'
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
    
    event_repr_name = f'stacked_histogram_dt={dt_ms}_nbins={num_bins}'
    event_repr_dir = output_path / 'event_representations_v2' / event_repr_name
    event_repr_dir.mkdir(parents=True, exist_ok=True)
    
    # 准备输出文件
    num_frames = len(disparity_timestamps)
    
    # 深度和掩码
    depth_maps = []
    depth_masks = []
    
    # 事件表示
    event_reprs = []
    
    # 时间戳
    timestamps_us = []
    
    # 文件路径
    events_file = input_path / 'interlaken_00_d_events_left' / 'events.h5'
    disparity_dir = input_path / 'interlaken_00_d_disparity_event'
    
    print(f"\n生成数据...")
    for idx in tqdm(range(num_frames), desc="处理帧"):
        # 加载视差图
        disparity_file = disparity_dir / f'{idx*2:06d}.png'  # DSEC使用偶数索引
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
        
        # 获取当前帧的时间戳
        current_time_ns = disparity_timestamps[idx]
        timestamps_us.append(current_time_ns // 1000)  # 转换为微秒
        
        # 加载对应时间窗口的事件
        dt_ns = dt_ms * 1_000_000  # 转换为纳秒
        start_time_ns = current_time_ns - dt_ns
        end_time_ns = current_time_ns
        
        try:
            x, y, t, p = load_events_in_time_window(
                events_file,
                start_time_ns,
                end_time_ns,
                max_events=500000  # 限制事件数量
            )
            
            # 生成事件表示
            event_repr = events_to_stacked_histogram(
                x, y, t, p,
                height=orig_height,
                width=orig_width,
                num_bins=num_bins
            )
            
            # 下采样事件表示
            if downsample:
                event_repr_tensor = torch.from_numpy(event_repr).unsqueeze(0)
                event_repr_tensor = torch.nn.functional.interpolate(
                    event_repr_tensor,
                    size=(target_height, target_width),
                    mode='nearest'
                )
                event_repr = event_repr_tensor.squeeze(0).numpy()
            
            event_reprs.append(event_repr)
        
        except Exception as e:
            print(f"警告: 处理事件数据失败 (帧{idx}): {e}")
            # 使用空事件表示
            event_repr = np.zeros((2 * num_bins, target_height, target_width), dtype=np.float32)
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
    
    print(f"  深度图: {depth_maps_array.shape} -> {depth_dir / f'depth_maps{ds_suffix}.h5'}")
    print(f"  掩码: {depth_masks_array.shape}")
    print(f"  有效深度比例: {depth_masks_array.mean()*100:.1f}%")
    
    # 保存事件表示
    print(f"\n保存事件表示...")
    event_reprs_array = np.array(event_reprs, dtype=np.float32)
    
    with h5py.File(event_repr_dir / f'event_representations{ds_suffix}.h5', 'w') as f:
        f.create_dataset('data', data=event_reprs_array, compression='gzip')
    
    # 创建索引映射
    objframe_idx_2_repr_idx = np.arange(len(event_reprs), dtype=np.int64)
    np.save(event_repr_dir / 'objframe_idx_2_repr_idx.npy', objframe_idx_2_repr_idx)
    np.save(event_repr_dir / 'timestamps_us.npy', np.array(timestamps_us))
    
    print(f"  事件表示: {event_reprs_array.shape} -> {event_repr_dir / f'event_representations{ds_suffix}.h5'}")
    print(f"\n✅ 序列 {sequence_name} 处理完成!")


def main():
    parser = argparse.ArgumentParser(description='处理DSEC数据集')
    parser.add_argument('--input_dir', type=str, default='data/DSEC/00_d',
                        help='DSEC原始数据目录')
    parser.add_argument('--output_dir', type=str, default='data/dsec_processed',
                        help='输出目录')
    parser.add_argument('--sequence_name', type=str, default='train_seq_000',
                        help='序列名称')
    parser.add_argument('--dt_ms', type=int, default=50,
                        help='事件累积时间窗口(毫秒)')
    parser.add_argument('--num_bins', type=int, default=10,
                        help='事件表示bins数量')
    parser.add_argument('--no_downsample', action='store_true',
                        help='不下采样到320x240')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("DSEC数据集预处理")
    print("=" * 70)
    
    process_dsec_sequence(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        sequence_name=args.sequence_name,
        dt_ms=args.dt_ms,
        num_bins=args.num_bins,
        downsample=not args.no_downsample,
    )
    
    print("\n" + "=" * 70)
    print("✅ 所有处理完成!")
    print("=" * 70)


if __name__ == '__main__':
    main()


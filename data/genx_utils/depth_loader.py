"""
Depth data loader extension for event camera depth estimation
This module provides utilities to load depth ground truth alongside event representations

USAGE:
To use this for your depth dataset, you need to organize your data as follows:

Dataset structure:
.
├── event_representations_v2
│   └── ev_representation_name
│       ├── event_representations.h5  (existing event data)
│       ├── objframe_idx_2_repr_idx.npy
│       └── timestamps_us.npy
├── depth_v2  (NEW - add this for depth estimation)
│   ├── depth_maps.h5  (depth ground truth in meters)
│   ├── depth_masks.h5  (valid depth masks)
│   └── timestamps_us.npy
└── labels_v2  (existing labels dir, not used for depth)

The depth_maps.h5 should contain:
- Dataset 'data' with shape (N, H, W) where N is number of frames
- Depth values in meters (will be converted to log space automatically)

The depth_masks.h5 should contain:
- Dataset 'data' with shape (N, H, W) with boolean or 0/1 values
- True/1 indicates valid depth, False/0 indicates invalid
"""
from pathlib import Path
from typing import Optional, Tuple
import h5py
import numpy as np
import torch


class DepthLoader:
    """
    Loads depth ground truth data for event camera depth estimation
    """
    def __init__(
        self,
        path: Path,
        downsample_by_factor_2: bool = False,
        min_depth: float = 0.1,
        max_depth: float = 100.0,
    ):
        """
        Args:
            path: Path to sequence directory
            downsample_by_factor_2: Whether depth is downsampled by factor 2
            min_depth: Minimum valid depth value (meters)
            max_depth: Maximum valid depth value (meters)
        """
        self.path = path
        self.min_depth = min_depth
        self.max_depth = max_depth
        
        depth_dir = path / 'depth_v2'
        assert depth_dir.is_dir(), f"Depth directory not found: {depth_dir}. Please create it and add depth data."
        
        ds_factor_str = '_ds2_nearest' if downsample_by_factor_2 else ''
        self.depth_file = depth_dir / f'depth_maps{ds_factor_str}.h5'
        self.mask_file = depth_dir / f'depth_masks{ds_factor_str}.h5'
        
        assert self.depth_file.exists(), f"Depth file not found: {self.depth_file}"
        assert self.mask_file.exists(), f"Mask file not found: {self.mask_file}"
        
        # Verify file structure
        with h5py.File(str(self.depth_file), 'r') as f:
            assert 'data' in f, "depth_maps.h5 must contain 'data' dataset"
            self.num_depth_frames = f['data'].shape[0]
        
        with h5py.File(str(self.mask_file), 'r') as f:
            assert 'data' in f, "depth_masks.h5 must contain 'data' dataset"
            assert f['data'].shape[0] == self.num_depth_frames
    
    def get_depth_and_mask(
        self,
        idx: int,
        convert_to_log: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load depth map and mask at given index
        
        Args:
            idx: Frame index
            convert_to_log: Whether to convert depth to log space
        
        Returns:
            depth: Depth map (1, H, W) in log space if convert_to_log=True
            mask: Valid depth mask (1, H, W) as boolean tensor
        """
        with h5py.File(str(self.depth_file), 'r') as f:
            depth = f['data'][idx]
        
        with h5py.File(str(self.mask_file), 'r') as f:
            mask = f['data'][idx]
        
        # Convert to torch tensors
        depth = torch.from_numpy(depth).float()
        mask = torch.from_numpy(mask).bool()
        
        # Add channel dimension
        depth = depth.unsqueeze(0)  # (1, H, W)
        mask = mask.unsqueeze(0)  # (1, H, W)
        
        # Clamp depth to valid range
        depth = torch.clamp(depth, self.min_depth, self.max_depth)
        
        # Create valid depth mask
        valid_mask = mask & (depth >= self.min_depth) & (depth <= self.max_depth)
        
        # Convert to log space for training
        if convert_to_log:
            depth = torch.log(depth)
        
        return depth, valid_mask
    
    def get_depth_sequence(
        self,
        start_idx: int,
        end_idx: int,
        convert_to_log: bool = True
    ) -> Tuple[list, list]:
        """
        Load a sequence of depth maps and masks
        
        Args:
            start_idx: Start frame index
            end_idx: End frame index (exclusive)
            convert_to_log: Whether to convert depth to log space
        
        Returns:
            depths: List of depth tensors (1, H, W)
            masks: List of mask tensors (1, H, W)
        """
        depths = []
        masks = []
        
        for idx in range(start_idx, end_idx):
            depth, mask = self.get_depth_and_mask(idx, convert_to_log)
            depths.append(depth)
            masks.append(mask)
        
        return depths, masks


def create_dummy_depth_data(
    output_path: Path,
    num_frames: int,
    height: int = 480,
    width: int = 640,
    downsample_by_factor_2: bool = False
):
    """
    Create dummy depth data for testing
    
    Args:
        output_path: Path to sequence directory
        num_frames: Number of frames to generate
        height: Image height
        width: Image width
        downsample_by_factor_2: Whether to downsample
    """
    if downsample_by_factor_2:
        height //= 2
        width //= 2
    
    depth_dir = output_path / 'depth_v2'
    depth_dir.mkdir(exist_ok=True, parents=True)
    
    ds_factor_str = '_ds2_nearest' if downsample_by_factor_2 else ''
    
    # Create dummy depth maps (random values between 1 and 50 meters)
    depth_maps = np.random.uniform(1.0, 50.0, (num_frames, height, width)).astype(np.float32)
    
    # Create dummy masks (80% valid pixels)
    depth_masks = np.random.rand(num_frames, height, width) > 0.2
    
    # Save to HDF5
    with h5py.File(str(depth_dir / f'depth_maps{ds_factor_str}.h5'), 'w') as f:
        f.create_dataset('data', data=depth_maps, compression='gzip')
    
    with h5py.File(str(depth_dir / f'depth_masks{ds_factor_str}.h5'), 'w') as f:
        f.create_dataset('data', data=depth_masks, compression='gzip')
    
    print(f"Created dummy depth data at {depth_dir}")
    print(f"  - depth_maps: {depth_maps.shape}")
    print(f"  - depth_masks: {depth_masks.shape}")


# Example of how to integrate DepthLoader into existing sequence classes:
"""
# In sequence_base.py or sequence_rnd.py, add to __init__:

from data.genx_utils.depth_loader import DepthLoader

class SequenceBase(MapDataPipe):
    def __init__(self, ..., load_depth: bool = False):
        # ... existing initialization ...
        
        # Add depth loader if needed
        self.load_depth = load_depth
        if self.load_depth:
            self.depth_loader = DepthLoader(
                path=path,
                downsample_by_factor_2=downsample_by_factor_2
            )

# In __getitem__ or data loading method, add:

def load_sample(self, idx):
    # ... existing event repr loading ...
    
    data = {
        DataType.EV_REPR: ev_repr,
        # ... other data ...
    }
    
    if self.load_depth:
        depth, mask = self.depth_loader.get_depth_and_mask(idx)
        data[DataType.DEPTH] = depth
        data[DataType.DEPTH_MASK] = mask
    
    return data
"""


from pathlib import Path
from typing import List, Optional, Union, Tuple, Dict, Any

import h5py
try:
    import hdf5plugin
except ImportError:
    pass
import numpy as np
import torch
from torch.utils.data import IterableDataset

from data.genx_utils.labels import SparselyBatchedObjectLabels
from data.genx_utils.sequence_base import SequenceBase, get_objframe_idx_2_repr_idx
from data.utils.augmentor import RandomSpatialAugmentorGenX
from data.utils.types import DataType, DatasetType, LoaderDataDictGenX
from utils.timers import TimerDummy as Timer


def _scalar_as_1d_array(scalar: Union[int, float]):
    return np.atleast_1d(scalar)


def _get_ev_repr_range_indices(indices: np.ndarray, max_len: int) -> List[Tuple[int, int]]:
    """
    Computes a list of index ranges based on the input array of indices and a maximum length.
    The index ranges are computed such that the difference between consecutive indices
    should not exceed the maximum length (max_len).

    Parameters:
    -----------
    indices : np.ndarray
        A NumPy array of indices, where the indices are sorted in ascending order.
    max_len : int
        The maximum allowed length between consecutive indices.

    Returns:
    --------
    out : List[Tuple[int, int]]
        A list of tuples, where each tuple contains two integers representing the start and
        stop indices of the range.
    """
    meta_indices_stop = np.flatnonzero(np.diff(indices) > max_len)

    meta_indices_start = np.concatenate((np.atleast_1d(0), meta_indices_stop + 1))
    meta_indices_stop = np.concatenate((meta_indices_stop, np.atleast_1d(len(indices) - 1)))

    out = list()
    for meta_idx_start, meta_idx_stop in zip(meta_indices_start, meta_indices_stop):
        idx_start = max(indices[meta_idx_start] - max_len + 1, 0)
        idx_stop = indices[meta_idx_stop] + 1
        out.append((idx_start, idx_stop))
    return out


class SequenceForIter(SequenceBase):
    def __init__(self,
                 path: Path,
                 ev_representation_name: str,
                 sequence_length: int,
                 dataset_type: DatasetType,
                 downsample_by_factor_2: bool,
                 range_indices: Optional[Tuple[int, int]] = None,
                 load_depth: bool = False,
                 min_depth: float = 0.1,
                 max_depth: float = 100.0):
        super().__init__(path=path,
                         ev_representation_name=ev_representation_name,
                         sequence_length=sequence_length,
                         dataset_type=dataset_type,
                         downsample_by_factor_2=downsample_by_factor_2,
                         only_load_end_labels=False,
                         load_depth=load_depth,
                         min_depth=min_depth,
                         max_depth=max_depth)

        with h5py.File(str(self.ev_repr_file), 'r') as h5f:
            num_ev_repr = h5f['data'].shape[0]
        if range_indices is None:
            repr_idx_start = max(self.objframe_idx_2_repr_idx[0] - sequence_length + 1, 0)
            repr_idx_stop = num_ev_repr
        else:
            repr_idx_start, repr_idx_stop = range_indices
        # Set start idx such that the first label is no further than the last timestamp of the first sample sub-sequence
        min_start_repr_idx = max(self.objframe_idx_2_repr_idx[0] - sequence_length + 1, 0)
        assert 0 <= min_start_repr_idx <= repr_idx_start < repr_idx_stop <= num_ev_repr, \
            f'{min_start_repr_idx=}, {repr_idx_start=}, {repr_idx_stop=}, {num_ev_repr=}, {path=}'

        self.start_indices = list(range(repr_idx_start, repr_idx_stop, sequence_length))
        self.stop_indices = self.start_indices[1:] + [repr_idx_stop]
        self.length = len(self.start_indices)

        self._padding_representation = None

    @staticmethod
    def get_sequences_with_guaranteed_labels(
            path: Path,
            ev_representation_name: str,
            sequence_length: int,
            dataset_type: DatasetType,
            downsample_by_factor_2: bool,
            load_depth: bool = False,
            min_depth: float = 0.1,
            max_depth: float = 100.0) -> List['SequenceForIter']:
        """Generate sequences such that we do always have labels within each sample of the sequence
        This is required for training such that we are guaranteed to always have labels in the training step.
        However, for validation we don't require this if we catch the special case.
        """
        objframe_idx_2_repr_idx = get_objframe_idx_2_repr_idx(
            path=path, ev_representation_name=ev_representation_name)
        # max diff for repr idx is sequence length
        range_indices_list = _get_ev_repr_range_indices(indices=objframe_idx_2_repr_idx, max_len=sequence_length)
        sequence_list = list()
        for range_indices in range_indices_list:
            sequence_list.append(
                SequenceForIter(path=path,
                                ev_representation_name=ev_representation_name,
                                sequence_length=sequence_length,
                                dataset_type=dataset_type,
                                downsample_by_factor_2=downsample_by_factor_2,
                                range_indices=range_indices,
                                load_depth=load_depth,
                                min_depth=min_depth,
                                max_depth=max_depth)
            )
        return sequence_list

    @property
    def padding_representation(self) -> torch.Tensor:
        if self._padding_representation is None:
            ev_repr = self._get_event_repr_torch(start_idx=0, end_idx=1)[0]
            self._padding_representation = torch.zeros_like(ev_repr)
        return self._padding_representation

    def get_fully_padded_sample(self) -> LoaderDataDictGenX:
        is_first_sample = False
        is_padded_mask = [True] * self.seq_len
        ev_repr = [self.padding_representation] * self.seq_len
        labels = [None] * self.seq_len
        sparse_labels = SparselyBatchedObjectLabels(sparse_object_labels_batch=labels)
        out = {
            DataType.EV_REPR: ev_repr,
            DataType.OBJLABELS_SEQ: sparse_labels,
            DataType.IS_FIRST_SAMPLE: is_first_sample,
            DataType.IS_PADDED_MASK: is_padded_mask,
        }
        return out

    def __len__(self):
        return self.length

    def __getitem__(self, index: int) -> LoaderDataDictGenX:
        start_idx = self.start_indices[index]
        end_idx = self.stop_indices[index]

        # sequence info ###
        sample_len = end_idx - start_idx
        assert self.seq_len >= sample_len > 0, f'{self.seq_len=}, {sample_len=}, {start_idx=}, {end_idx=}, ' \
                                               f'\n{self.start_indices=}\n{self.stop_indices=}'

        is_first_sample = True if index == 0 else False
        is_padded_mask = [False] * sample_len
        ###################

        # event representations ###
        with Timer(timer_name='read ev reprs'):
            ev_repr = self._get_event_repr_torch(start_idx=start_idx, end_idx=end_idx)
        assert len(ev_repr) == sample_len
        ###########################

        # labels ###
        labels = list()
        for repr_idx in range(start_idx, end_idx):
            labels.append(self._get_labels_from_repr_idx(repr_idx))
        assert len(labels) == len(ev_repr)
        ############

        # apply padding (if necessary) ###
        if sample_len < self.seq_len:
            padding_len = self.seq_len - sample_len

            is_padded_mask.extend([True] * padding_len)
            ev_repr.extend([self.padding_representation] * padding_len)
            labels.extend([None] * padding_len)
        ##################################

        # convert labels to sparse labels for datapipes and dataloader
        sparse_labels = SparselyBatchedObjectLabels(sparse_object_labels_batch=labels)

        out = {
            DataType.EV_REPR: ev_repr,
            DataType.OBJLABELS_SEQ: sparse_labels,
            DataType.IS_FIRST_SAMPLE: is_first_sample,
            DataType.IS_PADDED_MASK: is_padded_mask,
        }
        
        # Load depth data if available
        if self.load_depth and self.depth_loader is not None:
            with Timer(timer_name='read depth'):
                depths, masks = self.depth_loader.get_depth_sequence(
                    start_idx=start_idx,
                    end_idx=end_idx,
                    convert_to_log=True
                )
                # Apply padding if necessary
                if sample_len < self.seq_len:
                    padding_len = self.seq_len - sample_len
                    # Add zero depth and invalid mask for padded frames
                    padding_depth = torch.zeros_like(depths[0])
                    padding_mask = torch.zeros_like(masks[0])
                    depths.extend([padding_depth] * padding_len)
                    masks.extend([padding_mask] * padding_len)
                
                out[DataType.DEPTH] = depths
                out[DataType.DEPTH_MASK] = masks
        
        return out


class RandAugmentIterDataPipe(IterableDataset):
    def __init__(self, source_dp: IterableDataset, dataset_config: Dict[str, Any]):
        super().__init__()
        self.source_dp = source_dp

        resolution_hw = tuple(dataset_config['resolution_hw'])
        assert len(resolution_hw) == 2
        ds_by_factor_2 = dataset_config['downsample_by_factor_2']
        if ds_by_factor_2:
            resolution_hw = tuple(x // 2 for x in resolution_hw)

        # 仅在流式训练管线中启用水平翻转（不使用缩放/旋转）
        augm_config = dataset_config.get('data_augmentation', {})
        if isinstance(augm_config, dict):
            stream_augm_cfg = augm_config.get('stream', {})
        else:
            stream_augm_cfg = {}

        self.spatial_augmentor = RandomSpatialAugmentorGenX(
            dataset_hw=resolution_hw,
            automatic_randomization=False,
            augm_config=stream_augm_cfg,
        )

    def __iter__(self):
        # 对整条流随机一次增强参数，然后对该 worker 的所有样本复用（仅水平翻转）
        self.spatial_augmentor.randomize_augmentation()
        for x in self.source_dp:
            yield self.spatial_augmentor(x)

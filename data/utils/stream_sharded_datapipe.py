from typing import Any, List, Optional

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, IterableDataset

from data.utils.datapipe_compat import (
    ConcatIterableDataset,
    IterableWrapperDataset,
    MapToIterAdapter,
    ZipIterableDataset,
    ZipLongestIterableDataset,
)


class ShardedStreamingDataPipe(IterableDataset):
    def __init__(self, datapipe_list: List[Dataset], batch_size: int, fill_value: Optional[Any] = None):
        super().__init__()
        assert batch_size > 0

        # We require map-style Datasets instead of IterableDatasets because IterableDatasets must be deepcopied
        # in each worker. Instead, map-style Datasets can be converted to IterableDatasets in each worker
        # without requiring a deepcopy.
        # Note: Sorting is a heuristic to get potentially better distribution of workloads than taking the data as is.
        # Sort iterators from long to short.
        self.datapipe_list = sorted(datapipe_list, key=lambda x: len(x), reverse=True)
        self.batch_size = batch_size
        self.fill_value = fill_value

    @staticmethod
    def yield_pyramid_indices(start_idx: int, end_idx: int):
        while True:
            for idx in range(start_idx, end_idx):
                yield idx
            for idx in range(end_idx - 1, start_idx - 1, -1):
                yield idx

    @classmethod
    def assign_datapipes_to_worker(cls,
                                   sorted_datapipe_list: List[Dataset],
                                   total_num_workers: int,
                                   global_worker_id: int) -> List[Dataset]:
        num_datapipes = len(sorted_datapipe_list)
        assert num_datapipes >= total_num_workers > global_worker_id, \
            f'{num_datapipes=}, {total_num_workers=}, {global_worker_id=}'
        datapipes = []
        # Assumes sorted datapipes from long to short.
        global_worker_id_generator = cls.yield_pyramid_indices(start_idx=0, end_idx=total_num_workers)
        for idx, dp in enumerate(sorted_datapipe_list):
            generated_global_worker_id = next(global_worker_id_generator)
            if generated_global_worker_id == global_worker_id:
                datapipes.append(dp)
        assert len(sorted_datapipe_list) > 0
        return datapipes

    def get_zipped_stream_from_worker_datapipes(
            self, datapipe_list: List[Dataset], batch_size: int) -> ZipLongestIterableDataset:
        num_datapipes = len(datapipe_list)
        assert num_datapipes > 0
        assert batch_size > 0
        effective_batch_size = min(num_datapipes, batch_size)
        datapipe_list = sorted(datapipe_list, key=lambda x: len(x), reverse=True)
        zipped_streams = [[] for _ in range(effective_batch_size)]
        batch_id_generator = self.yield_pyramid_indices(start_idx=0, end_idx=effective_batch_size)
        for datapipe in datapipe_list:
            batch_idx = next(batch_id_generator)
            zipped_streams[batch_idx].append(datapipe)
        for idx, streams in enumerate(zipped_streams):
            zipped_streams[idx] = ConcatIterableDataset(*(MapToIterAdapter(stream) for stream in streams))
        zipped_streams = ZipLongestIterableDataset(*zipped_streams, fill_value=self.fill_value)
        return zipped_streams

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        local_worker_id = 0 if worker_info is None else worker_info.id
        local_num_workers = 1 if worker_info is None else worker_info.num_workers
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            global_rank = dist.get_rank()
        else:
            world_size = 1
            global_rank = 0
        total_num_workers = local_num_workers * world_size
        global_worker_id = global_rank * local_num_workers + local_worker_id

        local_datapipes = self.assign_datapipes_to_worker(sorted_datapipe_list=self.datapipe_list,
                                                          total_num_workers=total_num_workers,
                                                          global_worker_id=global_worker_id)
        zipped_stream = self.get_zipped_stream_from_worker_datapipes(datapipe_list=local_datapipes,
                                                                     batch_size=self.batch_size)
        # We also stream the local worker id for the use-case where we have a recurrent neural network that saves
        # its state based on the local worker id. We don't need the global worker id for that because the states
        # are saved in each DDP process (per GPU) separately and do not to communicate with each other.

        worker_id_stream = IterableWrapperDataset([local_worker_id]).cycle(count=None)
        zipped_stream = ZipIterableDataset(zipped_stream, worker_id_stream)

        return iter(zipped_stream)

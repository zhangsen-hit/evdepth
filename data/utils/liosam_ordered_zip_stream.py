"""
LiOSAM 训练用：按 contiguous run 分组，多 worker / DDP 分片后，将 run 轮询分配到 B 条并行流，
流内时间顺序与 ConcatStreamingDataPipe 一致（无全局随机打散），再 Zip 成 batch，供跨 batch RNN。

若某 worker 上分到的 run 数少于 batch_size，则无法填满 B 条流，会抛错（见 __iter__）。
"""

from typing import List, Type, Optional

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, IterableDataset

from data.utils.datapipe_compat import (
    ConcatIterableDataset,
    IterableWrapperDataset,
    MapToIterAdapter,
    ZipIterableDataset,
)


class LiosamOrderedZipStreamingDataPipe(IterableDataset):
    """
    :param grouped_runs: 全局顺序的 List[List[LiosamSequenceForIter]]，
        外层每个元素是一条 contiguous run（内层窗口按时间顺序）。
    """

    def __init__(
        self,
        grouped_runs: List[List[Dataset]],
        batch_size: int,
        augmentation_pipeline: Optional[Type[IterableDataset]],
    ):
        super().__init__()
        assert batch_size > 0
        assert len(grouped_runs) >= 0
        self.grouped_runs = grouped_runs
        self.batch_size = batch_size
        self.augmentation_dp = augmentation_pipeline if augmentation_pipeline is not None else _DummyIterable

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        local_worker_id = 0 if worker_info is None else worker_info.id
        local_num_workers = 1 if worker_info is None else worker_info.num_workers

        if dist.is_available() and dist.is_initialized():
            global_rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            global_rank = 0
            world_size = 1

        total_workers = max(1, world_size * local_num_workers)
        global_worker_id = global_rank * local_num_workers + local_worker_id

        runs_all = self.grouped_runs
        my_runs = runs_all[global_worker_id::total_workers]
        batch_size = self.batch_size

        if len(my_runs) > 0 and len(my_runs) < batch_size:
            raise RuntimeError(
                f"[liosam] This DataLoader worker has {len(my_runs)} contiguous runs but stream batch_size is "
                f"{batch_size}. Each parallel stream needs at least one run. Reduce `batch_size`, reduce "
                f"`num_workers`/DDP procs, or add more sequences. "
                f"(global_worker_id={global_worker_id}, total_workers={total_workers})"
            )

        if len(my_runs) == 0:
            # 该 shard 无 run；返回空迭代
            return iter(())

        streams: List[ConcatIterableDataset] = []
        for b in range(batch_size):
            runs_for_stream_b = [my_runs[i] for i in range(len(my_runs)) if i % batch_size == b]
            aug_chain = []
            for run in runs_for_stream_b:
                for win_ds in run:
                    aug_chain.append(self.augmentation_dp(MapToIterAdapter(win_ds)))

            if not aug_chain:
                raise RuntimeError(
                    f"[liosam] Stream {b=} is empty after run assignment — cannot zip. batch_size={batch_size}, "
                    f"runs_on_worker={len(my_runs)}"
                )

            streams.append(ConcatIterableDataset(*aug_chain))

        zipped_samples = ZipIterableDataset(*streams)
        worker_id_stream = IterableWrapperDataset([local_worker_id]).cycle(count=None)
        return iter(ZipIterableDataset(zipped_samples, worker_id_stream))


class _DummyIterable(IterableDataset):
    """占位，与 ConcatStreamingDataPipe.DummyIterableDataset 相同语义。"""

    def __init__(self, source_dp: IterableDataset):
        super().__init__()
        self.source_dp = source_dp

    def __iter__(self):
        yield from self.source_dp

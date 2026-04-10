"""
Utilities for merging batches from mixed sampling (STREAM + RANDOM).

在 MIXED 采样模式下，我们会同时使用基于流式 DataPipe 的 dataloader（STREAM）
和基于随机访问的 dataloader（RANDOM）。本模块提供：

- mixed_collate_fn: 逐字段合并来自两个 dataloader 的数据（tensor / label / list）；
- merge_mixed_batches: 将 RANDOM / STREAM 两个 batch 合并为一个标准 batch。
"""

from typing import Any, Dict, List, Union

import torch as th

from data.genx_utils.labels import SparselyBatchedObjectLabels
from data.utils.types import DatasetSamplingMode


def mixed_collate_fn(x1: Union[th.Tensor, List[th.Tensor]], x2: Union[th.Tensor, List[th.Tensor]]):
    """
    用于 MIXED 采样模式：将来自 STREAM 和 RANDOM 两个 dataloader 的对应字段合并。

    - tensor: 在 batch 维拼接；
    - SparselyBatchedObjectLabels: 使用 `+` 合并；
    - list: 递归地对每个元素调用自身。
    """
    if isinstance(x1, th.Tensor):
        assert isinstance(x2, th.Tensor)
        return th.cat((x1, x2))
    if isinstance(x1, SparselyBatchedObjectLabels):
        assert isinstance(x2, SparselyBatchedObjectLabels)
        return x1 + x2
    if isinstance(x1, list):
        assert isinstance(x2, list)
        assert len(x1) == len(x2)
        return [mixed_collate_fn(x1=el_1, x2=el_2) for el_1, el_2 in zip(x1, x2)]
    raise NotImplementedError


def merge_mixed_batches(batch: Dict[str, Any]) -> Dict[str, Any]:
    """
    针对 MIXED 训练模式，将 RANDOM / STREAM 两个 dataloader 的 batch 合并为单一 batch。

    训练步骤中就可以像普通 batch 一样访问：
        batch['worker_id'], batch['data'][DataType.EV_REPR], ...
    """
    if "data" in batch:
        return batch
    rnd_data = batch[DatasetSamplingMode.RANDOM]["data"]
    stream_batch = batch[DatasetSamplingMode.STREAM]
    # 只关心 streaming dataloader 的 worker_id，因为 random dataloader 的状态每个 batch 都会重置。
    out = {"worker_id": stream_batch["worker_id"]}
    stream_data = stream_batch["data"]
    assert rnd_data.keys() == stream_data.keys(), f"{rnd_data.keys()=}, {stream_data.keys()=}"
    data_out: Dict[str, Any] = {}
    for key in rnd_data.keys():
        data_out[key] = mixed_collate_fn(stream_data[key], rnd_data[key])
    out.update({"data": data_out})
    return out



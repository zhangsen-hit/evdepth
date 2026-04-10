"""
PyTorch 原生替代 torchdata DataPipe 工具类。

提供与 torchdata.datapipes 中 Concater / Zipper / ZipperLongest /
IterableWrapper / MapDataPipe.to_iter_datapipe() 等价的轻量实现，
全部基于 torch.utils.data.Dataset / IterableDataset。
"""
from itertools import chain, zip_longest
from typing import Any, Optional

from torch.utils.data import Dataset, IterableDataset


class MapToIterAdapter(IterableDataset):
    """将 Dataset (map-style) 转换为 IterableDataset，等价于
    torchdata 中 MapDataPipe.to_iter_datapipe()。"""

    def __init__(self, map_dataset: Dataset):
        self.map_dataset = map_dataset

    def __iter__(self):
        for i in range(len(self.map_dataset)):
            yield self.map_dataset[i]


class ConcatIterableDataset(IterableDataset):
    """顺序串接多个可迭代数据集，等价于 torchdata.datapipes.iter.Concater。"""

    def __init__(self, *datasets):
        self.datasets = datasets

    def __iter__(self):
        return chain(*self.datasets)


class ZipIterableDataset(IterableDataset):
    """按位置 zip 多路可迭代数据集，等价于 torchdata.datapipes.iter.Zipper。
    最短流耗尽即停止。"""

    def __init__(self, *datasets):
        self.datasets = datasets

    def __iter__(self):
        return zip(*self.datasets)

    def zip(self, other):
        return ZipIterableDataset(self, other)


class ZipLongestIterableDataset(IterableDataset):
    """按位置 zip 多路可迭代数据集（最长对齐），
    等价于 torchdata.datapipes.iter.ZipperLongest。
    短流用 fill_value 补齐。"""

    def __init__(self, *datasets, fill_value: Optional[Any] = None):
        self.datasets = datasets
        self.fill_value = fill_value

    def __iter__(self):
        return zip_longest(*self.datasets, fillvalue=self.fill_value)

    def zip(self, other):
        return ZipIterableDataset(self, other)


class IterableWrapperDataset(IterableDataset):
    """将普通可迭代对象包装为 IterableDataset，
    等价于 torchdata.datapipes.iter.IterableWrapper。"""

    def __init__(self, iterable):
        self.iterable = iterable

    def __iter__(self):
        return iter(self.iterable)

    def cycle(self, count=None):
        return _CycleIterableDataset(self, count)


class _CycleIterableDataset(IterableDataset):
    """无限（或有限次）循环一个 IterableDataset。"""

    def __init__(self, source, count=None):
        self.source = source
        self.count = count

    def __iter__(self):
        if self.count is None:
            while True:
                yield from self.source
        else:
            for _ in range(self.count):
                yield from self.source

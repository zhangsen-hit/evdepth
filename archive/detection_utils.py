"""
Archive: detection-related utility classes (backbone / event representation selectors).

当前项目中这两个类没有被实际使用，仅作为备份保留：
- BackboneFeatureSelector: 用于跨 batch 收集并拼接 backbone 特征；
- EventReprSelector: 用于收集事件表示（event representations）并按区间取出。

如果未来确认不需要，可以直接删除本文件。
"""

from typing import List, Optional

import torch as th

from data.utils.types import BackboneFeatures


class BackboneFeatureSelector:
    def __init__(self):
        self.features = None
        self.reset()

    def reset(self):
        self.features = dict()

    def add_backbone_features(
        self,
        backbone_features: BackboneFeatures,
        selected_indices: Optional[List[int]] = None,
    ) -> None:
        if selected_indices is not None:
            assert len(selected_indices) > 0
        for k, v in backbone_features.items():
            if k not in self.features:
                self.features[k] = [v[selected_indices]] if selected_indices is not None else [v]
            else:
                self.features[k].append(v[selected_indices] if selected_indices is not None else v)

    def get_batched_backbone_features(self) -> Optional[BackboneFeatures]:
        if len(self.features) == 0:
            return None
        return {k: th.cat(v, dim=0) for k, v in self.features.items()}


class EventReprSelector:
    def __init__(self):
        self.repr_list = None
        self.reset()

    def reset(self):
        self.repr_list = list()

    def __len__(self):
        return len(self.repr_list)

    def add_event_representations(
        self, event_representations: th.Tensor, selected_indices: Optional[List[int]] = None
    ) -> None:
        if selected_indices is not None:
            assert len(selected_indices) > 0
        self.repr_list.extend(x[0] for x in event_representations[selected_indices].split(1))

    def get_event_representations_as_list(
        self, start_idx: int = 0, end_idx: Optional[int] = None
    ) -> Optional[List[th.Tensor]]:
        if len(self) == 0:
            return None
        if end_idx is None:
            end_idx = len(self)
        assert start_idx < end_idx, f"{start_idx=}, {end_idx=}"
        return self.repr_list[start_idx:end_idx]



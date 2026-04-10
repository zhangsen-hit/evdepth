"""
管理 RNN/LSTM 隐状态在 batch 之间的连续性。

典型用法场景：
- DataLoader 使用多个 worker 执行流式采样（STREAM / MIXED）；
- 同一 worker 在时间上连续地产生 batch；
- 需要在相邻 batch 之间复用 hidden state，同时在新序列起点处 reset。
"""

from typing import Dict, List, Optional, Tuple, Union

import torch as th

from data.utils.types import LstmStates


class RNNStates:
    """
    管理 RNN/LSTM 隐状态生命周期：
    - 按 worker_id 存取状态（兼容 DataPipe 多 worker）；
    - 支持递归 detach，避免梯度跨 batch 传播；
    - 支持按样本索引/布尔 mask 将部分状态置零（序列重置）。
    """

    def __init__(self):
        self.states: Dict[int, LstmStates] = {}

    def _has_states(self) -> bool:
        return len(self.states) > 0

    @classmethod
    def recursive_detach(cls, inp: Union[th.Tensor, List, Tuple, Dict]):
        if isinstance(inp, th.Tensor):
            return inp.detach()
        if isinstance(inp, list):
            return [cls.recursive_detach(x) for x in inp]
        if isinstance(inp, tuple):
            return tuple(cls.recursive_detach(x) for x in inp)
        if isinstance(inp, dict):
            return {k: cls.recursive_detach(v) for k, v in inp.items()}
        raise NotImplementedError

    @classmethod
    def recursive_reset(
        cls,
        inp: Union[th.Tensor, List, Tuple, Dict],
        indices_or_bool_tensor: Optional[Union[List[int], th.Tensor]] = None,
    ):
        if isinstance(inp, th.Tensor):
            assert inp.requires_grad is False, "Not assumed here but should be the case."
            if indices_or_bool_tensor is None:
                inp[:] = 0
            else:
                assert len(indices_or_bool_tensor) > 0
                inp[indices_or_bool_tensor] = 0
            return inp
        if isinstance(inp, list):
            return [cls.recursive_reset(x, indices_or_bool_tensor=indices_or_bool_tensor) for x in inp]
        if isinstance(inp, tuple):
            return tuple(cls.recursive_reset(x, indices_or_bool_tensor=indices_or_bool_tensor) for x in inp)
        if isinstance(inp, dict):
            return {k: cls.recursive_reset(v, indices_or_bool_tensor=indices_or_bool_tensor) for k, v in inp.items()}
        raise NotImplementedError

    def save_states_and_detach(self, worker_id: int, states: LstmStates) -> None:
        """保存某个 worker 最新的 hidden state，并递归 detach 以切断梯度。"""
        self.states[worker_id] = self.recursive_detach(states)

    def get_states(self, worker_id: int) -> Optional[LstmStates]:
        """获取某个 worker 上一次保存的 hidden state（可能为 None）。"""
        if not self._has_states():
            return None
        if worker_id not in self.states:
            return None
        return self.states[worker_id]

    def reset(self, worker_id: int, indices_or_bool_tensor: Optional[Union[List[int], th.Tensor]] = None):
        """
        将某个 worker 中指定样本的 hidden state 置零。

        - indices_or_bool_tensor 为 None：清零该 worker 所有样本的 state；
        - 为 bool tensor（shape=[B]）：True 的位置会被 reset；
        - 为 int list：这些 index 对应的样本会被 reset。
        """
        if not self._has_states():
            return
        if worker_id in self.states:
            self.states[worker_id] = self.recursive_reset(
                self.states[worker_id], indices_or_bool_tensor=indices_or_bool_tensor
            )



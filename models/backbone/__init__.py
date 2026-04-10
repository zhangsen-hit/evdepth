from typing import Dict, Any

from .mobilenet_rnn import MobileNetRNN
from .base import BaseDetector


def build_recurrent_backbone(backbone_cfg: Dict[str, Any]):
    """
    構建循環 backbone 的統一入口。
    原先位於 `models.detection.recurrent_backbone`，現移至 `models.backbone`。
    """
    name = backbone_cfg["name"]
    if name == "MobileNetRNN":
        return MobileNetRNN(backbone_cfg)
    raise NotImplementedError(
        f"Backbone '{name}' not implemented. Only 'MobileNetRNN' is available."
    )


__all__ = ["MobileNetRNN", "BaseDetector", "build_recurrent_backbone"]


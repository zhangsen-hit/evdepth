"""
FPN构建函数 - 用于深度估计

注意：YOLOXHead已移除，只保留FPN用于深度估计
"""
from typing import Tuple, Dict, Any

from .yolo_pafpn import YOLOPAFPN


def build_yolox_fpn(fpn_cfg: Dict[str, Any], in_channels: Tuple[int, ...]):
    """
    构建FPN网络
    
    Args:
        fpn_cfg: FPN配置
        in_channels: 输入通道数
    
    Returns:
        FPN网络实例
    """
    fpn_cfg_dict = dict(fpn_cfg)  # 创建副本以避免修改原配置
    fpn_name = fpn_cfg_dict.pop('name')
    fpn_cfg_dict.update({"in_channels": in_channels})
    if fpn_name in {'PAFPN', 'pafpn'}:
        compile_cfg = fpn_cfg_dict.pop('compile', None)
        fpn_cfg_dict.update({"compile_cfg": compile_cfg})
        return YOLOPAFPN(**fpn_cfg_dict)
    raise NotImplementedError(f"FPN类型 '{fpn_name}' 未实现")

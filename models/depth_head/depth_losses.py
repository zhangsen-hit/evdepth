"""
Depth estimation loss functions
Includes SILog (Scale-Invariant Log) and image gradient loss

重要说明：
- 训练数据与网络原始输出目前都是“log(depth)”标量空间；
- 本文件会在 loss 计算之前，把 `pred` 和 `target` 都转换到
  “归一化 log(depth)（min_depth=0.5, max_depth=80）”空间，再计算误差。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class BerHuLoss(nn.Module):
    """
    Reverse Huber (BerHu) loss
    Behaves like L1 for small errors and L2 for large errors
    """
    def __init__(self, threshold: float = 0.2):
        super().__init__()
        self.threshold = threshold
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            pred: predicted depth (normalized log space)
            target: ground truth depth (normalized log space)
            mask: valid depth mask
        """
        if mask is not None:
            pred = pred[mask]
            target = target[mask]

        # If there are no valid pixels, return zero loss to avoid max() on empty tensor
        if pred.numel() == 0 or target.numel() == 0:
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        diff = torch.abs(pred - target)
        delta = self.threshold * torch.max(diff).item()
        
        # BerHu: L1 for small errors, L2 for large errors
        berhu_loss = torch.where(
            diff <= delta,
            diff,
            (diff ** 2 + delta ** 2) / (2 * delta)
        )
        
        return berhu_loss.mean()


class SILogLoss(nn.Module):
    """
    Scale-Invariant Logarithmic loss
    Invariant to global scale of the scene
    """
    def __init__(self, lambd: float = 0.5):
        super().__init__()
        self.lambd = lambd
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            pred: predicted depth (normalized log space)
            target: ground truth depth (normalized log space)
            mask: valid depth mask
        """
        if mask is not None:
            pred = pred[mask]
            target = target[mask]

        # If there are no valid pixels, return zero loss
        if pred.numel() == 0 or target.numel() == 0:
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        diff = pred - target
        
        # Scale-invariant term
        loss = torch.mean(diff ** 2) - self.lambd * (torch.mean(diff) ** 2)
        
        return loss


class GradientLoss(nn.Module):
    """
    Image gradient loss to encourage smooth depth predictions
    Computes L1 loss on spatial gradients
    """
    def __init__(self):
        super().__init__()
    
    def gradient(self, x: torch.Tensor):
        """Compute spatial gradients"""
        # Gradient in x direction
        grad_x = x[:, :, :, :-1] - x[:, :, :, 1:]
        # Gradient in y direction
        grad_y = x[:, :, :-1, :] - x[:, :, 1:, :]
        return grad_x, grad_y
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            pred: predicted depth (B, 1, H, W) in normalized log space
            target: ground truth depth (B, 1, H, W) in normalized log space
            mask: valid depth mask (B, 1, H, W)
        """
        pred_grad_x, pred_grad_y = self.gradient(pred)
        target_grad_x, target_grad_y = self.gradient(target)

        if mask is not None:
            # Adjust mask for gradient dimensions
            mask_x = mask[:, :, :, :-1] * mask[:, :, :, 1:]
            mask_y = mask[:, :, :-1, :] * mask[:, :, 1:, :]

            diff_x = torch.abs(pred_grad_x - target_grad_x)
            diff_y = torch.abs(pred_grad_y - target_grad_y)

            # If there are no valid pixels for gradients, return zero loss
            if mask_x.sum() == 0 and mask_y.sum() == 0:
                return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

            loss_x = diff_x[mask_x].mean() if mask_x.any() else torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
            loss_y = diff_y[mask_y].mean() if mask_y.any() else torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        else:
            loss_x = torch.abs(pred_grad_x - target_grad_x).mean()
            loss_y = torch.abs(pred_grad_y - target_grad_y).mean()

        return loss_x + loss_y


class DepthLoss(nn.Module):
    """
    Combined multi-scale depth loss
    Combines SILog and gradient losses at multiple scales
    """
    def __init__(
        self,
        silog_weight: float = 1.0,
        grad_weight: float = 0.5,
        silog_lambda: float = 0.5,
        scales: list = [2, 4, 8, 16],  # Multi-scale outputs
        depth_min: float = 0.5,
        depth_max: float = 80.0,
    ):
        super().__init__()
        self.silog_weight = silog_weight
        self.grad_weight = grad_weight
        self.scales = scales
        self.depth_min = float(depth_min)
        self.depth_max = float(depth_max)

        if self.depth_min <= 0 or self.depth_max <= 0:
            raise ValueError(f"depth_min/depth_max must be > 0, got {depth_min=} {depth_max=}")
        if self.depth_min >= self.depth_max:
            raise ValueError(f"depth_min must be < depth_max, got {depth_min=} {depth_max=}")

        # Precompute constants for log(depth)->norm log(depth) transform
        self._log_depth_min = math.log(self.depth_min)
        self._log_depth_max = math.log(self.depth_max)
        self._log_depth_denom = self._log_depth_max - self._log_depth_min  # > 0

        self.silog_loss = SILogLoss(lambd=silog_lambda)
        self.grad_loss = GradientLoss()

    def log_depth_to_norm_log_depth(self, log_depth: torch.Tensor) -> torch.Tensor:
        """
        兼容旧流程：log(depth) -> 归一化 log 空间(norm_log)。

        当前 B 方案下，网络输出与 GT 目标已被预处理到 norm_log，
        loss 不再对 pred 做该变换，函数仅保留以兼容其它调用/调试。
        """
        log_depth_clamped = torch.clamp(log_depth, min=self._log_depth_min, max=self._log_depth_max)
        return (log_depth_clamped - self._log_depth_min) / max(self._log_depth_denom, 1e-6)

    # def forward(self, predictions: dict, target: torch.Tensor, mask: torch.Tensor = None):
    #     """
    #     Args:
    #         predictions: dict with keys like 'depth_2', 'depth_4', 'depth_8', 'depth_16'
    #         target: ground truth depth (B, 1, H, W) in log space
    #         mask: valid depth mask (B, 1, H, W)
        
    #     Returns:
    #         dict of losses
    #     """
    #     total_loss = 0.0
    #     losses_dict = {}
        
    #     # Process each scale
    #     for scale in self.scales:
    #         key = f'depth_{scale}'
    #         if key not in predictions:
    #             continue
            
    #         pred = predictions[key]
            
    #         # Downsample target and mask to match prediction scale
    #         target_scaled = F.interpolate(
    #             target, size=pred.shape[-2:], mode='bilinear', align_corners=False
    #         )
            
    #         if mask is not None:
    #             mask_scaled = F.interpolate(
    #                 mask.float(), size=pred.shape[-2:], mode='nearest'
    #             ).bool()
    #         else:
    #             mask_scaled = None
            
    #         # Compute individual losses
    #         berhu = self.berhu_loss(pred, target_scaled, mask_scaled)
    #         silog = self.silog_loss(pred, target_scaled, mask_scaled)
    #         grad = self.grad_loss(pred, target_scaled, mask_scaled)
            
    #         # Weighted combination
    #         scale_loss = (
    #             self.berhu_weight * berhu +
    #             self.silog_weight * silog +
    #             self.grad_weight * grad
    #         )
            
    #         # Weight loss by scale (higher weight for finer scales)
    #         scale_weight = 1.0 / scale  # Higher resolution gets more weight
    #         total_loss += scale_weight * scale_loss
            
    #         # Store individual losses for logging
    #         losses_dict[f'berhu_{scale}'] = berhu.item()
    #         losses_dict[f'silog_{scale}'] = silog.item()
    #         losses_dict[f'grad_{scale}'] = grad.item()
    #         losses_dict[f'loss_{scale}'] = scale_loss.item()
        
    #     # Normalize by number of scales
    #     total_loss = total_loss / len(self.scales)
    #     losses_dict['loss'] = total_loss
        
    #     return total_loss, losses_dict

    def forward(self, predictions, target, mask=None):
        """
        多尺度深度 loss（稀疏 GT 友好版本）。

        关键原则（参考 Monodepth2）：不下采样稀疏 GT，而是把各尺度预测统一上采样到 GT
        的原始分辨率，只在原始高分辨率 mask 的有效点上计算 loss。这样做的好处：
        1) 不丢失本就稀疏的激光雷达有效点；
        2) 不把“无效值”随下采样扩散到更大范围（否则近地面的无效区会越来越大，
           与天空成片无效混淆，导致近地面被错估为远处并出现条纹状伪影）；
        3) 低分辨率层也能受到原分辨率 GT 的高频精确监督。
        """
        total_loss = torch.zeros((), device=target.device, dtype=target.dtype)
        losses_dict = {}

        # GT 和 mask 固定在输入（高）分辨率，全程不下采样
        target_hw = target.shape[-2:]
        target_norm_log = torch.clamp(target, 0.0, 1.0)
        if mask is not None and mask.dtype != torch.bool:
            mask = mask.bool()

        for scale in self.scales:
            key = f'depth_{scale}'
            if key not in predictions:
                continue

            pred = predictions[key]

            # 双线性上采样预测到 GT 的原始分辨率（不是把 GT 降到 pred 分辨率）
            if pred.shape[-2:] != target_hw:
                pred_up = F.interpolate(
                    pred, size=target_hw, mode='bilinear', align_corners=False
                )
            else:
                pred_up = pred

            # pred/target 均已在 norm_log 空间，直接计算
            pred_norm_log = torch.clamp(pred_up, 0.0, 1.0)

            silog = self.silog_loss(pred_norm_log, target_norm_log, mask)
            grad = self.grad_loss(pred_norm_log, target_norm_log, mask)
            scale_loss = self.silog_weight * silog + self.grad_weight * grad

            scale_weight = 1.0 / scale
            if scale == 2:
                scale_weight *= 1.0  # 给最高分辨率的输出更高权重
            total_loss += scale_weight * scale_loss

            # 仅用于日志的子项，提前 detach，避免构建过大的计算图
            losses_dict[f'silog_{scale}'] = silog.detach()
            losses_dict[f'grad_{scale}'] = grad.detach()

        # 总损失必须保持梯度信息，不能 detach
        losses_dict['loss'] = total_loss
        return total_loss, losses_dict

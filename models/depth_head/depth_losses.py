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

    可选：远距像素软加权
        w_i = 1 + alpha * clamp((t_i - t0) / (1 - t0), 0, 1)
    其中 t_i 为 target 在 norm_log 空间的值（∈[0,1]）。
    alpha=0 时严格等价于未加权版本。权重只依赖 target，且会 detach，不参与反传。
    """
    def __init__(self, lambd: float = 0.5, far_weight_alpha: float = 0.0, far_weight_t0: float = 0.3):
        super().__init__()
        self.lambd = lambd
        self.far_weight_alpha = float(far_weight_alpha)
        self.far_weight_t0 = float(far_weight_t0)
        if not (0.0 <= self.far_weight_t0 < 1.0):
            raise ValueError(f"far_weight_t0 must be in [0, 1), got {far_weight_t0}")

    def _far_weight(self, target_norm_log: torch.Tensor) -> torch.Tensor:
        """基于 target 的 per-pixel 远距权重；alpha=0 时为全 1。"""
        if self.far_weight_alpha == 0.0:
            return torch.ones_like(target_norm_log)
        denom = max(1.0 - self.far_weight_t0, 1e-6)
        w = 1.0 + self.far_weight_alpha * torch.clamp(
            (target_norm_log - self.far_weight_t0) / denom, 0.0, 1.0
        )
        return w.detach()

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

        if self.far_weight_alpha == 0.0:
            loss = torch.mean(diff ** 2) - self.lambd * (torch.mean(diff) ** 2)
            return loss

        # 加权版 SILog：用 target 作为远近度量计算权重，保持尺度不变项形式
        w = self._far_weight(target)
        w_sum = w.sum().clamp(min=1e-6)
        mean_wdiff2 = (w * diff ** 2).sum() / w_sum
        mean_wdiff = (w * diff).sum() / w_sum
        loss = mean_wdiff2 - self.lambd * (mean_wdiff ** 2)
        return loss


class MultiScaleGradientLoss(nn.Module):
    """
    E2Depth 风格的多尺度梯度匹配损失 (Hidalgo-Carrio et al., 3DV 2020)。

    注意：这里的 "多尺度" 指的是**同一张最终 residual 的不同下采样尺度**，
    而不是 decoder 的多个输出头。公式：

        R = pred_final - target
        R^s = downsample^s(R)      # s = 0, 1, 2, 3
        L_grad = (1 / n_valid) * sum_s sum_u ( |∂x R^s(u)| + |∂y R^s(u)| )

    作用是鼓励深度平滑、同时保留清晰的深度不连续边界。

    稀疏 GT / mask 处理：
    - residual 与 mask 同步 avg_pool2d 下采样；
    - 下采样过程中使用 "只对有效像素求均值" 的方式，避免无效像素把残差人为拉向 0；
    - 梯度项只在 **相邻两格都有效** 的位置计入损失；
    - 某尺度有效对数过少时，不会触发除零。
    """

    def __init__(self, num_scales: int = 4, mask_threshold: float = 0.0):
        super().__init__()
        self.num_scales = int(num_scales)
        # 下采样后一个 2x2 块的有效占比严格大于该阈值，才视为该粗尺度块有效
        # 0.0 表示 "只要块内有一个有效像素就保留"，对稀疏 LiDAR 比较友好
        self.mask_threshold = float(mask_threshold)

    @staticmethod
    def _gradient_xy(x: torch.Tensor):
        grad_x = x[:, :, :, :-1] - x[:, :, :, 1:]
        grad_y = x[:, :, :-1, :] - x[:, :, 1:, :]
        return grad_x, grad_y

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            pred:   最终（最高分辨率）深度预测，(B, 1, H, W)，与 target 在同一空间
                    （当前项目是 norm_log 空间）。
            target: GT 深度，(B, 1, H, W)，同上空间。
            mask:   有效像素 mask，(B, 1, H, W) bool/0-1，可为 None。
        """
        if pred.shape != target.shape:
            raise ValueError(
                f"pred and target must have same shape, got {pred.shape} vs {target.shape}"
            )

        residual = pred - target

        if mask is not None:
            mask_f = mask.to(residual.dtype)
            # 把无效像素处的 residual 置 0；后续用 mask 归一化，等价于 "对有效像素求均值"
            cur_res = residual * mask_f
            cur_mask = mask_f
        else:
            cur_res = residual
            cur_mask = None

        total_loss = torch.zeros((), device=pred.device, dtype=pred.dtype)
        total_count = torch.zeros((), device=pred.device, dtype=pred.dtype)

        for s in range(self.num_scales):
            if s > 0:
                # —— 同步下采样 residual 和 mask ——
                if cur_mask is not None:
                    weighted = F.avg_pool2d(cur_res, kernel_size=2, stride=2)
                    mask_avg = F.avg_pool2d(cur_mask, kernel_size=2, stride=2)
                    # 仅对块内有效像素求均值：sum(res*mask)/sum(mask)
                    cur_res = weighted / mask_avg.clamp(min=1e-6)
                    cur_mask = (mask_avg > self.mask_threshold).to(cur_res.dtype)
                    cur_res = cur_res * cur_mask  # 无效块再置 0，防止数值泄漏
                else:
                    cur_res = F.avg_pool2d(cur_res, kernel_size=2, stride=2)

            # 防御：太小的 feature map 无法做差分
            if cur_res.shape[-1] < 2 or cur_res.shape[-2] < 2:
                break

            grad_x, grad_y = self._gradient_xy(cur_res)

            if cur_mask is not None:
                # 只有相邻两格都有效，对应位置的梯度才计入
                mx = cur_mask[:, :, :, :-1] * cur_mask[:, :, :, 1:]
                my = cur_mask[:, :, :-1, :] * cur_mask[:, :, 1:, :]
                total_loss = total_loss + (grad_x.abs() * mx).sum() + (grad_y.abs() * my).sum()
                total_count = total_count + mx.sum() + my.sum()
            else:
                total_loss = total_loss + grad_x.abs().sum() + grad_y.abs().sum()
                total_count = total_count + torch.tensor(
                    float(grad_x.numel() + grad_y.numel()),
                    device=pred.device, dtype=pred.dtype,
                )

        # 整体按有效像素数归一化；有效像素过少时 clamp 到 1 可避免除零
        # （此时 total_loss 本身也近似为 0，不会制造虚假梯度）
        return total_loss / total_count.clamp(min=1.0)


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
        far_weight_alpha: float = 0.0,
        far_weight_t0: float = 0.3,
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

        self.silog_loss = SILogLoss(
            lambd=silog_lambda,
            far_weight_alpha=far_weight_alpha,
            far_weight_t0=far_weight_t0,
        )
        # E2Depth 风格：gradient loss 只作用在最终（最高分辨率）预测的 residual 上，
        # 并在 residual 的多个下采样尺度上累计（默认 4 个尺度）。
        self.grad_loss = MultiScaleGradientLoss(num_scales=4)

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
        2) 不把"无效值"随下采样扩散到更大范围（否则近地面的无效区会越来越大，
           与天空成片无效混淆，导致近地面被错估为远处并出现条纹状伪影）；
        3) 低分辨率层也能受到原分辨率 GT 的高频精确监督。

        E2Depth 修正（重要）：
        - SILog 主损失仍在多个 decoder 输出尺度上监督（auxiliary supervision，不变）。
        - gradient loss 不再对每个 decoder 头分别计算，而是只对最终（最高分辨率）预测
          `depth_2` 计算一次 E2Depth 风格多尺度 residual gradient loss。
        """
        total_loss = torch.zeros((), device=target.device, dtype=target.dtype)
        losses_dict = {}

        # GT 和 mask 固定在输入（高）分辨率，全程不下采样
        target_hw = target.shape[-2:]
        target_norm_log = torch.clamp(target, 0.0, 1.0)
        if mask is not None and mask.dtype != torch.bool:
            mask = mask.bool()

        # ---- (1) SILog 主损失：保留原有多尺度 decoder 头监督 ----
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
            scale_loss = self.silog_weight * silog

            scale_weight = 1.0 / scale  # 高分辨率头权重更大
            total_loss = total_loss + scale_weight * scale_loss

            losses_dict[f'silog_{scale}'] = silog.detach()

        # ---- (2) E2Depth 多尺度 gradient loss：仅作用于最终最高分辨率预测 ----
        # 约定最精细 decoder 输出头（scales 里最小的那个，一般是 depth_2）作为 pred_final。
        finest_scale = min(self.scales) if len(self.scales) > 0 else None
        finest_key = f'depth_{finest_scale}' if finest_scale is not None else None
        if finest_key is not None and finest_key in predictions:
            pred_final = predictions[finest_key]
            if pred_final.shape[-2:] != target_hw:
                pred_final = F.interpolate(
                    pred_final, size=target_hw, mode='bilinear', align_corners=False
                )
            pred_final_norm_log = torch.clamp(pred_final, 0.0, 1.0)

            grad_multiscale = self.grad_loss(
                pred_final_norm_log, target_norm_log, mask
            )
            total_loss = total_loss + self.grad_weight * grad_multiscale
            losses_dict['grad_multiscale'] = grad_multiscale.detach()

        # 总损失必须保持梯度信息，不能 detach
        losses_dict['loss'] = total_loss
        return total_loss, losses_dict

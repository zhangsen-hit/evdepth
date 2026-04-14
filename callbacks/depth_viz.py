"""
Depth Estimation Visualization Callback
Visualizes depth predictions, ground truth, and event representations
"""
from enum import Enum, auto
from typing import Any, Dict

import numpy as np
import torch as th
import math
import os
import matplotlib.pyplot as plt

from pytorch_lightning.utilities.rank_zero import rank_zero_only

from callbacks.viz_base import VizCallbackBase
from loggers.wandb_logger import WandbLogger
from modules.depth_estimation import DepthOutput
from data.utils.types import DataType


class DepthVizBufferEntry(Enum):
    """Buffer entries for depth visualization"""
    DEPTH_PRED = auto()  # Predicted depth
    DEPTH_GT = auto()    # Ground truth depth
    EV_REPR = auto()     # Event representation
    DEPTH_MASK = auto()  # GT valid mask (True=valid), invalid will be black in viz


class DepthVizCallback(VizCallbackBase):
    """
    Callback for visualizing depth estimation results
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config=config, buffer_entries=DepthVizBufferEntry)
        # 标记当前 epoch 是否已经为训练集保存过可视化（只保存第一个 batch）
        self._train_epoch_viz_done: bool = False
        # 测试 epoch 可视化标记（只保存第一个 batch）
        self._test_epoch_viz_done: bool = False
        # 第一个 epoch 中用于 check_input 的全局样本序号（每 batch 每样本递增）
        self._check_input_counter: int = 0

        # 深度显示范围（真实深度，meters）
        model_depth_range = config.get("model", {}).get("depth_range", {})
        dataset_depth_range = config.get("dataset", {}).get("depth_range", {})
        depth_range = model_depth_range if isinstance(model_depth_range, dict) and len(model_depth_range) > 0 else dataset_depth_range
        self.depth_min = float(depth_range.get("min", 0.5))
        self.depth_max = float(depth_range.get("max", 80.0))

    def _log_depth_to_norm_log(self, log_depth: th.Tensor) -> th.Tensor:
        """
        将 log(depth) 映射到归一化 log 空间：
            norm_log = (log(depth_clamped) - log(depth_min)) / (log(depth_max) - log(depth_min))
        """
        log_depth_f = log_depth.to(dtype=th.float32)
        finite = th.isfinite(log_depth_f)

        log_min = math.log(self.depth_min)
        log_max = math.log(self.depth_max)
        log_depth_clamped = th.clamp(log_depth_f, min=log_min, max=log_max)
        log_depth_clamped = th.where(finite, log_depth_clamped, th.zeros_like(log_depth_clamped))

        return (log_depth_clamped - log_min) / max(log_max - log_min, 1e-6)

    def _norm_log_to_depth_real(self, norm_log: th.Tensor) -> th.Tensor:
        """归一化 log(depth) -> 真实深度（meters）"""
        norm_log_f = norm_log.to(dtype=th.float32)
        log_min = math.log(self.depth_min)
        log_max = math.log(self.depth_max)
        log_depth_clamped = norm_log_f * (log_max - log_min) + log_min
        return th.exp(log_depth_clamped)

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        # 每个 epoch 开始时重置标记，这样只会在该 epoch 的第一个 batch 上保存可视化
        self._train_epoch_viz_done = False
        # 第一个 epoch 开始时创建 check_input 并重置序号
        if trainer.current_epoch == 0:
            self._check_input_counter = 0
            if trainer.is_global_zero:
                os.makedirs("check_input", exist_ok=True)

    @rank_zero_only
    def on_train_batch_end(
            self,
            trainer,
            pl_module,
            outputs: Any,
            batch: Any,
            batch_idx: int,
            dataloader_idx: int = 0,
    ) -> None:
        # 先调用基类逻辑：根据配置把训练可视化发到 wandb
        super().on_train_batch_end(
            trainer=trainer,
            pl_module=pl_module,
            outputs=outputs,
            batch=batch,
            batch_idx=batch_idx,
            dataloader_idx=dataloader_idx,
        )

        # 第一个 epoch：每个 batch 的每组 input 与 GT 写入 check_input/，用于检查输入与数据增强
        # 放在这里，不受 _train_epoch_viz_done 限制，这样 epoch 0 的所有 batch 都会被保存。
        self._save_check_input(trainer, outputs, batch, dataloader_idx)

        # 只对主 dataloader 的第一个 batch 做一次可视化
        if dataloader_idx != 0:
            return
        if self._train_epoch_viz_done:
            return
        if outputs is None or outputs.get(DepthOutput.SKIP_VIZ, False):
            return

        # 预测：使用 Module 输出的 norm_log 预测（与训练/评估一致）
        depth_pred = outputs[DepthOutput.DEPTH_PRED]  # (B, 1, H_pred, W_pred) in norm_log
        ev_repr = outputs[DepthOutput.EV_REPR]        # (B, C, H, W) 可视化用事件图

        # GT：为了与 check_input / 验证路径一致，这里直接从 batch 原始 DataType.DEPTH 中取
        data = batch.get("data", batch)
        if DataType.DEPTH not in data:
            return
        depth_seq = data[DataType.DEPTH]              # list[T] or tensor, log(depth)
        depth_mask_seq = data.get(DataType.DEPTH_MASK, None)
        # 取序列最后一帧（与 training_step / val 可视化一致）
        if isinstance(depth_seq, (list, tuple)):
            depth_last = depth_seq[-1]
        else:
            depth_last = depth_seq
        if depth_last.dim() == 3:
            depth_last = depth_last.unsqueeze(1)      # (B,1,H,W)
        depth_last = depth_last.to(dtype=depth_pred.dtype, device=depth_pred.device)

        if depth_mask_seq is not None:
            if isinstance(depth_mask_seq, (list, tuple)):
                depth_mask_last = depth_mask_seq[-1]
            else:
                depth_mask_last = depth_mask_seq
            if depth_mask_last.dim() == 3:
                depth_mask_last = depth_mask_last.unsqueeze(1)  # (B,1,H,W)
            depth_mask_last = depth_mask_last.to(dtype=th.bool, device=depth_pred.device)
        else:
            depth_mask_last = None

        # 将 GT / mask 对齐到预测分辨率，仅用于可视化，不影响训练逻辑
        target_hw = depth_pred.shape[-2:]
        if depth_last.shape[-2:] != target_hw:
            depth_last = th.nn.functional.interpolate(
                depth_last,
                size=target_hw,
                mode="bilinear",
                align_corners=False,
            )
        if depth_mask_last is not None and depth_mask_last.shape[-2:] != target_hw:
            depth_mask_last = th.nn.functional.interpolate(
                depth_mask_last.float(),
                size=target_hw,
                mode="nearest",
            ).bool()

        # 预测：norm_log -> real depth
        depth_pred_real = self._norm_log_to_depth_real(depth_pred.detach()).cpu().numpy()
        # GT：log(depth_raw) -> real depth
        depth_gt_real = np.exp(depth_last.detach().cpu().numpy().astype(np.float32))
        ev_repr_np = ev_repr.detach().cpu().numpy()
        depth_mask_np = None
        if depth_mask_last is not None:
            depth_mask_np = depth_mask_last.detach().cpu().numpy().astype(bool)

        B = depth_pred_real.shape[0]

        # 如果存在 padding 列（例如宽度 348，而事件图宽度是 346），
        # 为了与事件图完全对齐，可视化时裁掉右侧多余列，消除伪影竖条。
        H_d, W_d = depth_pred_real.shape[-2:]
        _, _, H_ev, W_ev = ev_repr_np.shape
        if W_d != W_ev:
            depth_pred_real = depth_pred_real[..., :W_ev]
            depth_gt_real = depth_gt_real[..., :W_ev]
            if depth_mask_np is not None:
                depth_mask_np = depth_mask_np[..., :W_ev]

        root_dir = "depth_epoch_viz_train"
        epoch_dir = os.path.join(root_dir, f"epoch_{trainer.current_epoch:06d}")
        os.makedirs(epoch_dir, exist_ok=True)

        for b in range(B):
            mask_b = None if depth_mask_np is None else depth_mask_np[b, 0]
            pred_img = self._depth_to_colormap(
                depth_pred_real[b, 0],
                vmin=self.depth_min,
                vmax=self.depth_max,
                mask=mask_b,
                is_error=False,
            )
            # 额外保存一份“全有效像素”的预测深度图（不使用 GT mask）
            pred_img_full = self._depth_to_colormap(
                depth_pred_real[b, 0],
                vmin=self.depth_min,
                vmax=self.depth_max,
                mask=None,
                is_error=False,
            )
            gt_img = self._depth_to_colormap(
                depth_gt_real[b, 0],
                vmin=self.depth_min,
                vmax=self.depth_max,
                mask=mask_b,
                is_error=False,
            )
            ev_img = self.ev_repr_to_img(ev_repr_np[b])

            # 误差图（与验证阶段一致，只是这里没有显式 mask，全图参与）
            error = np.abs(depth_pred_real[b, 0] - depth_gt_real[b, 0])
            error_img = self._depth_to_colormap(
                error,
                vmin=0.0,
                vmax=5.0,
                mask=mask_b,
                is_error=True,
            )

            base_name = f"sample{b:02d}"
            plt.imsave(os.path.join(epoch_dir, f"{base_name}_pred.png"), pred_img)
            plt.imsave(os.path.join(epoch_dir, f"{base_name}_predfull.png"), pred_img_full)
            plt.imsave(os.path.join(epoch_dir, f"{base_name}_gt.png"), gt_img)
            plt.imsave(os.path.join(epoch_dir, f"{base_name}_ev.png"), ev_img)
            plt.imsave(os.path.join(epoch_dir, f"{base_name}_err.png"), error_img)

        # 标记本 epoch 已写入，避免后续 batch 重复写
        self._train_epoch_viz_done = True

    @rank_zero_only
    def _save_check_input(self, trainer, outputs: Any, batch: Any, dataloader_idx: int) -> None:
        """仅在第一轮 epoch 中，将每个 batch 的输入（事件图）和 GT（深度）保存到 check_input/。"""
        if trainer.current_epoch != 0:
            return
        if dataloader_idx != 0:
            return
        if outputs is None or outputs.get(DepthOutput.SKIP_VIZ, False):
            return

        # === 使用 batch 中的原始数据（未 pad 或仅按数据管线 pad），在“原始空间分辨率”上检查输入 ===
        data = batch.get("data", batch)
        if DataType.EV_REPR not in data or DataType.DEPTH not in data:
            return

        ev_repr = data[DataType.EV_REPR]      # list[T] or tensor
        depth = data[DataType.DEPTH]          # list[T] or tensor, in log(depth)
        depth_mask = data.get(DataType.DEPTH_MASK, None)  # list[T] or tensor, bool

        # 只取序列中的最后一帧（与 training_step / val 可视化一致）
        if isinstance(ev_repr, (list, tuple)):
            ev_repr = ev_repr[-1]
        if isinstance(depth, (list, tuple)):
            depth = depth[-1]
        if depth_mask is not None and isinstance(depth_mask, (list, tuple)):
            depth_mask = depth_mask[-1]

        # 取 batch 内所有样本，ev_repr: (B,C,H,W)，depth: (B,1,H,W) or (B,H,W)
        if ev_repr.dim() == 4:
            pass
        else:
            # 不符合预期形状时直接跳过，避免崩溃
            return
        if depth.dim() == 3:
            depth = depth.unsqueeze(1)
        if depth_mask is not None and depth_mask.dim() == 3:
            depth_mask = depth_mask.unsqueeze(1)

        ev_repr_np = ev_repr.detach().cpu().numpy()
        depth_log_np = depth.detach().cpu().numpy().astype(np.float32)
        # 原始 DataType.DEPTH 是 log(depth)，直接 exp 还原到真实深度（meters）
        depth_gt_real = np.exp(depth_log_np)

        depth_mask_np = None
        if depth_mask is not None:
            depth_mask_np = depth_mask.detach().cpu().numpy().astype(bool)

        B = ev_repr_np.shape[0]
        out_dir = "check_input"
        for b in range(B):
            idx = self._check_input_counter
            input_img = self.ev_repr_to_img(ev_repr_np[b])
            mask_b = None if depth_mask_np is None else depth_mask_np[b, 0]
            gt_img = self._depth_to_colormap(
                depth_gt_real[b, 0],
                vmin=self.depth_min,
                vmax=self.depth_max,
                mask=mask_b,
                is_error=False,
            )
            plt.imsave(os.path.join(out_dir, f"{idx}_input.png"), input_img)
            plt.imsave(os.path.join(out_dir, f"{idx}_gt.png"), gt_img)
            self._check_input_counter += 1

    def on_train_batch_end_custom(self,
                                  logger: WandbLogger,
                                  outputs: Any,
                                  batch: Any,
                                  log_n_samples: int,
                                  global_step: int) -> None:
        """
        Log depth visualizations during training
        """
        if outputs.get(DepthOutput.SKIP_VIZ, False):
            return
        
        # 预测：使用 Module 输出的 norm_log 预测
        depth_pred = outputs[DepthOutput.DEPTH_PRED]  # (B, 1, H, W) in norm_log
        ev_repr = outputs[DepthOutput.EV_REPR]        # (B, C, H, W)

        # GT / mask：从 batch 原始 DataType.DEPTH / DEPTH_MASK 中取，保证与 check_input / 验证一致
        data = batch.get("data", batch)
        if DataType.DEPTH not in data:
            return
        depth_seq = data[DataType.DEPTH]              # list[T] or tensor, log(depth)
        depth_mask_seq = data.get(DataType.DEPTH_MASK, None)
        if isinstance(depth_seq, (list, tuple)):
            depth_last = depth_seq[-1]
        else:
            depth_last = depth_seq
        if depth_last.dim() == 3:
            depth_last = depth_last.unsqueeze(1)      # (B,1,H,W)
        depth_last = depth_last.to(dtype=depth_pred.dtype, device=depth_pred.device)

        if depth_mask_seq is not None:
            if isinstance(depth_mask_seq, (list, tuple)):
                depth_mask_last = depth_mask_seq[-1]
            else:
                depth_mask_last = depth_mask_seq
            if depth_mask_last.dim() == 3:
                depth_mask_last = depth_mask_last.unsqueeze(1)
            depth_mask_last = depth_mask_last.to(dtype=th.bool, device=depth_pred.device)
        else:
            depth_mask_last = None
        
        # Limit to log_n_samples
        depth_pred = depth_pred[:log_n_samples]
        ev_repr = ev_repr[:log_n_samples]
        depth_last = depth_last[:log_n_samples]
        if depth_mask_last is not None:
            depth_mask_last = depth_mask_last[:log_n_samples]

        # 将 GT / mask 对齐到预测分辨率，仅用于可视化
        target_hw = depth_pred.shape[-2:]
        if depth_last.shape[-2:] != target_hw:
            depth_last = th.nn.functional.interpolate(
                depth_last,
                size=target_hw,
                mode="bilinear",
                align_corners=False,
            )
        if depth_mask_last is not None and depth_mask_last.shape[-2:] != target_hw:
            depth_mask_last = th.nn.functional.interpolate(
                depth_mask_last.float(),
                size=target_hw,
                mode="nearest",
            ).bool()

        # 预测：norm_log -> real depth
        depth_pred_real = self._norm_log_to_depth_real(depth_pred.detach()).cpu().numpy()
        # GT：log(depth_raw) -> real depth
        depth_gt_real = np.exp(depth_last.detach().cpu().numpy().astype(np.float32))
        ev_repr_np = ev_repr.detach().cpu().numpy()
        depth_mask_np = None
        if depth_mask_last is not None:
            depth_mask_np = depth_mask_last.detach().cpu().numpy().astype(bool)
        
        # Create visualizations并使用 WandbLogger.log_images 规范记录图像
        pred_imgs = []
        gt_imgs = []
        ev_imgs = []
        err_imgs = []
        for i in range(depth_pred_real.shape[0]):
            mask_i = None if depth_mask_np is None else depth_mask_np[i, 0]
            # Convert depth to colormap
            pred_img = self._depth_to_colormap(
                depth_pred_real[i, 0],
                vmin=self.depth_min,
                vmax=self.depth_max,
                mask=mask_i,
                is_error=False,
            )
            gt_img = self._depth_to_colormap(
                depth_gt_real[i, 0],
                vmin=self.depth_min,
                vmax=self.depth_max,
                mask=mask_i,
                is_error=False,
            )
            ev_img = self.ev_repr_to_img(ev_repr_np[i])

            pred_imgs.append(pred_img)
            gt_imgs.append(gt_img)
            ev_imgs.append(ev_img)
            error = np.abs(depth_pred_real[i, 0] - depth_gt_real[i, 0])
            error_img = self._depth_to_colormap(
                error,
                vmin=0.0,
                vmax=5.0,
                mask=mask_i,
                is_error=True,
            )
            err_imgs.append(error_img)

        # 使用专门的 log_images 接口，内部会包装为 wandb.Image，避免 ndarray 序列化 Warning
        if len(pred_imgs) > 0:
            logger.log_images("train/depth_pred", pred_imgs, step=global_step)
            logger.log_images("train/depth_gt", gt_imgs, step=global_step)
            logger.log_images("train/events", ev_imgs, step=global_step)
            logger.log_images("train/error", err_imgs, step=global_step)
    
    def on_validation_batch_end_custom(self,
                                       batch: Any,
                                       outputs: Any) -> None:
        """
        Collect data for validation visualization
        """
        if outputs is None or outputs.get(DepthOutput.SKIP_VIZ, False):
            return
        
        # Collect depth predictions, ground truth, events, and optional GT mask
        depth_pred = outputs[DepthOutput.DEPTH_PRED]  # (T, 1, H, W)
        depth_gt = outputs[DepthOutput.DEPTH_GT]
        ev_repr = outputs[DepthOutput.EV_REPR]
        depth_mask = outputs.get(DepthOutput.DEPTH_VIZ_MASK)  # (T, H, W) or None
        # Buffer 不能存 None：无 mask 时用全 True（表示没有已知的无效区域）
        if depth_mask is None:
            T, _, H, W = depth_gt.shape
            # 如果没有提供深度 mask，则视为“未知无效区域”，不强行将其当作无效（避免整张图变黑）
            depth_mask = th.ones(T, H, W, dtype=th.bool, device=depth_gt.device)

        self.add_to_buffer(DepthVizBufferEntry.DEPTH_PRED, depth_pred)
        self.add_to_buffer(DepthVizBufferEntry.DEPTH_GT, depth_gt)
        self.add_to_buffer(DepthVizBufferEntry.EV_REPR, ev_repr)
        self.add_to_buffer(DepthVizBufferEntry.DEPTH_MASK, depth_mask)
    
    def on_validation_epoch_end_custom(self,
                                       logger: WandbLogger,
                                       global_step: int,
                                       epoch: int,
                                       stage: str = "val") -> None:
        """
        Log validation visualizations at epoch end
        """
        depth_preds = self.get_from_buffer(DepthVizBufferEntry.DEPTH_PRED)
        depth_gts = self.get_from_buffer(DepthVizBufferEntry.DEPTH_GT)
        ev_reprs = self.get_from_buffer(DepthVizBufferEntry.EV_REPR)
        depth_masks = self.get_from_buffer(DepthVizBufferEntry.DEPTH_MASK)  # list of (T,H,W) or None per batch

        if len(depth_preds) == 0:
            return

        # Stack and convert from norm_log to real space
        depth_preds = th.stack(depth_preds)  # (N, T, 1, H, W)
        depth_gts = th.stack(depth_gts)      # (N, T, 1, H, W)
        ev_reprs = th.stack(ev_reprs)        # (N, T, C, H, W)

        depth_preds_real = self._norm_log_to_depth_real(depth_preds.detach()).cpu().numpy()
        depth_gts_real = self._norm_log_to_depth_real(depth_gts.detach()).cpu().numpy()
        ev_reprs_np = ev_reprs.detach().cpu().numpy()
        # masks: list of length N, each (T,H,W) or None
        mask_np_list = []
        for m in depth_masks:
            if m is not None:
                mask_np_list.append(m.cpu().numpy())
            else:
                mask_np_list.append(None)

        N, T = depth_preds_real.shape[0], depth_preds_real.shape[1]

        # Create visualizations，使用 WandbLogger.log_images 记录
        pred_imgs_last = []
        gt_imgs_last = []
        ev_imgs_last = []
        err_imgs_last = []

        # === 每个 epoch 建立一个子文件夹，保存整个序列 ===
        root_dir = "depth_epoch_viz" if stage == "val" else f"depth_epoch_viz_{stage}"
        epoch_dir = os.path.join(root_dir, f"epoch_{epoch:06d}")
        os.makedirs(epoch_dir, exist_ok=True)

        for n in range(N):
            for t in range(T):
                pred_img = self._depth_to_colormap(
                    depth_preds_real[n, t, 0],
                    vmin=self.depth_min,
                    vmax=self.depth_max,
                    mask=mask_np_list[n][t],
                    is_error=False,
                )
                # 额外保存一份“全有效像素”的预测深度图（不使用 GT mask）
                pred_img_full = self._depth_to_colormap(
                    depth_preds_real[n, t, 0],
                    vmin=self.depth_min,
                    vmax=self.depth_max,
                    mask=None,
                    is_error=False,
                )
                gt_img = self._depth_to_colormap(
                    depth_gts_real[n, t, 0],
                    mask=mask_np_list[n][t],
                    vmin=self.depth_min,
                    vmax=self.depth_max,
                    is_error=False,
                )
                ev_img = self.ev_repr_to_img(ev_reprs_np[n, t])

                # Error map: only on valid GT pixels; invalid → black
                error = np.abs(depth_preds_real[n, t, 0] - depth_gts_real[n, t, 0])
                error_img = self._depth_to_colormap(
                    error,
                    vmin=0.0,
                    vmax=5.0,
                    mask=mask_np_list[n][t],
                    is_error=True,
                )

                # 保存到本地：预测 / GT / 事件 / 误差图
                base_name = f"sample{n:02d}_t{t:03d}"
                plt.imsave(os.path.join(epoch_dir, f"{base_name}_pred.png"), pred_img)
                plt.imsave(os.path.join(epoch_dir, f"{base_name}_predfull.png"), pred_img_full)
                plt.imsave(os.path.join(epoch_dir, f"{base_name}_gt.png"), gt_img)
                plt.imsave(os.path.join(epoch_dir, f"{base_name}_ev.png"), ev_img)
                plt.imsave(os.path.join(epoch_dir, f"{base_name}_err.png"), error_img)

            # 供 wandb 日志使用：每个样本只记录最后一帧，避免太多图片
            pred_imgs_last.append(
                self._depth_to_colormap(
                    depth_preds_real[n, T - 1, 0],
                    vmin=self.depth_min,
                    vmax=self.depth_max,
                    mask=mask_np_list[n][T - 1],
                    is_error=False,
                )
            )
            gt_imgs_last.append(
                self._depth_to_colormap(
                    depth_gts_real[n, T - 1, 0],
                    mask=mask_np_list[n][T - 1],
                    vmin=self.depth_min,
                    vmax=self.depth_max,
                    is_error=False,
                )
            )
            ev_imgs_last.append(
                self.ev_repr_to_img(ev_reprs_np[n, T - 1])
            )
            error_last = np.abs(
                depth_preds_real[n, T - 1, 0] - depth_gts_real[n, T - 1, 0]
            )
            err_imgs_last.append(
                self._depth_to_colormap(
                    error_last,
                    vmin=0.0,
                    vmax=5.0,
                    mask=mask_np_list[n][T - 1],
                    is_error=True,
                )
            )

        if len(pred_imgs_last) > 0:
            stage_prefix = "val" if stage == "val" else stage
            logger.log_images(f"{stage_prefix}/depth_pred", pred_imgs_last, step=global_step)
            logger.log_images(f"{stage_prefix}/depth_gt", gt_imgs_last, step=global_step)
            logger.log_images(f"{stage_prefix}/events", ev_imgs_last, step=global_step)
            logger.log_images(f"{stage_prefix}/error", err_imgs_last, step=global_step)

    @rank_zero_only
    def on_test_epoch_start(self, trainer, pl_module) -> None:
        self._reset_buffer()
        self._test_epoch_viz_done = False

    @rank_zero_only
    def on_test_batch_end(
        self,
        trainer,
        pl_module,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        # 复用 validation.high_dim 配置：同样决定是否开启可视化、以及采样的 batch 数
        log_test_hd = self.log_config["validation"]["high_dim"]
        if not log_test_hd.get("enable", False):
            return
        if dataloader_idx != 0:
            return
        if self._test_epoch_viz_done:
            return
        if outputs is None or outputs.get(DepthOutput.SKIP_VIZ, False):
            return

        self.on_validation_batch_end_custom(batch, outputs)
        self._test_epoch_viz_done = True

    @rank_zero_only
    def on_test_epoch_end(self, trainer, pl_module) -> None:
        log_test_hd = self.log_config["validation"]["high_dim"]
        if not log_test_hd.get("enable", False):
            return
        if not self._test_epoch_viz_done:
            return
        logger = trainer.logger
        if logger is None:
            return
        assert isinstance(logger, WandbLogger)
        self.on_validation_epoch_end_custom(
            logger=logger,
            global_step=trainer.global_step,
            epoch=trainer.current_epoch,
            stage="test",
        )
    
    @staticmethod
    def _depth_to_colormap(
        depth: np.ndarray,
        vmin: float = None,
        vmax: float = None,
        mask: np.ndarray = None,
        is_error: bool = False,
    ) -> np.ndarray:
        """
        深度/误差图转 colormap（jet + log 空间 + 红近蓝远 + 无效黑色）。

        颜色规则：
        - 红色：表示“近”（值小，log 空间归一化接近低端）
        - 蓝色：表示“远”（值大，log 空间归一化接近高端）
        - 黑色：无效值（来自 mask，或非有限值；深度图额外要求值>0）

        其中深度图的输入是 meters（真实深度），误差图的输入是 meters 差值（>=0）。
        """
        import matplotlib.cm as cm

        values = np.asarray(depth, dtype=np.float32)
        h, w = values.shape
        colormap = cm.get_cmap("jet")

        out = np.zeros((h, w, 3), dtype=np.uint8)  # 无效处保持纯黑
        eps = 1e-6

        if mask is None:
            valid = np.isfinite(values)
            if not is_error:
                valid = valid & (values > 0)
            if not valid.any():
                return out

            vmin_use = float(np.min(values[valid])) if vmin is None else float(vmin)
            vmax_use = float(np.max(values[valid])) if vmax is None else float(vmax)
            if vmax_use <= vmin_use:
                return out

            vmin_eff = max(vmin_use, eps)  # log 空间要求 >0
            vmax_eff = max(vmax_use, vmin_eff + eps)
            log_vmin = math.log(vmin_eff)
            log_vmax = math.log(vmax_eff)
            denom = max(log_vmax - log_vmin, 1e-12)

            log_values = np.log(np.clip(values, vmin_eff, None))
            norm = np.clip((log_values - log_vmin) / denom, 0.0, 1.0)

            # jet 默认：蓝(低) -> 红(高)；我们要“红近蓝远”，因此反转归一化。
            norm_rev = 1.0 - norm
            colored = (colormap(norm_rev)[:, :, :3] * 255).astype(np.uint8)
            out[valid] = colored[valid]
            return out

        # mask 路径：只给有效像素上色，其余保持黑色
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (h, w):
            mask = np.broadcast_to(mask.reshape(-1), (h, w)).reshape(h, w)
        valid = mask & np.isfinite(values)
        if not is_error:
            valid = valid & (values > 0)
        if not valid.any():
            return out

        vmin_use = float(np.min(values[valid])) if vmin is None else float(vmin)
        vmax_use = float(np.max(values[valid])) if vmax is None else float(vmax)
        if vmax_use <= vmin_use:
            return out

        vmin_eff = max(vmin_use, eps)
        vmax_eff = max(vmax_use, vmin_eff + eps)
        log_vmin = math.log(vmin_eff)
        log_vmax = math.log(vmax_eff)
        denom = max(log_vmax - log_vmin, 1e-12)

        log_values_valid = np.log(np.clip(values[valid], vmin_eff, None))
        norm_valid = np.clip((log_values_valid - log_vmin) / denom, 0.0, 1.0)
        norm_valid_rev = 1.0 - norm_valid

        out[valid] = (colormap(norm_valid_rev)[:, :3] * 255).astype(np.uint8)
        return out


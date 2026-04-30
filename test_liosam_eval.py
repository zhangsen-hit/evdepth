#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LiOSAM 测试集评估：指定场景（默认 22、23），逐帧连续推理（约 5ms / 帧），
汇总 delta1 / RMSE 写入 local_loss，并在 test_viz 下导出序列 PNG 与 MP4。
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, Tuple

import numpy as np
import torch
import torch.nn.functional as F

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from callbacks.depth_viz import DepthVizCallback  # noqa: E402
from data.genx_utils.liosam_sequence import (  # noqa: E402
    _load_depth_and_mask_from_npz,
    find_contiguous_runs,
    load_liosam_index,
)
from data.utils.types import DataType  # noqa: E402
from modules.depth_estimation import Module as DepthModule  # noqa: E402
from modules.data.rnn_states_across_batches import RNNStates  # noqa: E402
from utils.evaluation.depth import DepthEvaluator  # noqa: E402

def _finest_depth_pred(predictions: Dict[str, torch.Tensor]) -> torch.Tensor:
    return predictions.get("depth_1", predictions["depth_2"])


def _extract_liosam_dt(cfg: Dict[str, Any]) -> Tuple[float, float]:
    fi = float(cfg.get("frame_interval_sec", 0.005))
    dev = float(cfg.get("max_interval_deviation_sec", 0.003))
    return fi - dev, fi + dev


def iter_liosam_scene_frames(
    scene_path: Path,
    dataset_config: Dict[str, Any],
) -> Iterator[Tuple[bool, Dict[str, Any]]]:
    """
    按时间顺序逐帧遍历某场景目录下所有「时间连续 run」内的帧。
    每个 run 的第一帧 yield is_first_sample=True（用于 RNN reset）。
    """
    entries = load_liosam_index(scene_path)
    if not entries:
        return
    min_dt, max_dt = _extract_liosam_dt(dataset_config)
    runs = find_contiguous_runs(entries, min_dt_sec=min_dt, max_dt_sec=max_dt)
    ev_key = dataset_config.get("ev_key", "input")
    depth_key = dataset_config.get("depth_key", "label")
    depth_mask_key = dataset_config.get("depth_mask_key", None)
    dr = dataset_config.get("depth_range", {})
    min_depth = float(dr.get("min", 0.5))
    max_depth = float(dr.get("max", 80.0))

    for run in runs:
        for pos, entry_idx in enumerate(run):
            _, _, filename = entries[entry_idx]
            fn = scene_path / filename
            data = np.load(str(fn), allow_pickle=True)
            ev = data[ev_key]
            ev_t = torch.from_numpy(np.asarray(ev, dtype=np.float32))
            if ev_t.dim() == 2:
                ev_t = ev_t.unsqueeze(0)

            depth_t, mask_t = _load_depth_and_mask_from_npz(
                data, depth_key, depth_mask_key, min_depth, max_depth, convert_to_log=True
            )

            payload = {
                "ev_repr": ev_t,
                "depth_log": depth_t,
                "depth_mask": mask_t,
            }
            is_first = pos == 0
            yield is_first, payload


def _collate_like_streaming(sample: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "data": {
            DataType.EV_REPR: [sample["ev_repr"]],
            DataType.DEPTH: [sample["depth_log"]],
            DataType.DEPTH_MASK: [sample["depth_mask"]],
            DataType.IS_FIRST_SAMPLE: torch.tensor([sample["is_first"]], dtype=torch.bool),
        },
        "worker_id": 0,
    }


def load_model_weights(model: DepthModule, ckpt_path: Path) -> None:
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    result = model.load_state_dict(state, strict=False)
    print(f"[ckpt] missing={result.missing_keys[:5]}... n={len(result.missing_keys)}")
    print(f"[ckpt] unexpected={result.unexpected_keys[:5]}... n={len(result.unexpected_keys)}")


def write_local_loss_metrics(delta1: float, rmse: float) -> None:
    os.makedirs("local_loss", exist_ok=True)
    with open(os.path.join("local_loss", "test_delta1.txt"), "w", encoding="utf-8") as f:
        f.write(f"0\t{delta1}\n")
    with open(os.path.join("local_loss", "test_rmse.txt"), "w", encoding="utf-8") as f:
        f.write(f"0\t{rmse}\n")
    print(f"[local_loss] 已写入 test_delta1.txt / test_rmse.txt（格式与 train_loss 一致：索引\\t数值）")


def images_to_mp4(frame_dir: Path, out_mp4: Path, fps: float) -> None:
    ff = shutil.which("ffmpeg")
    if ff is None:
        print("[warn] 未找到 ffmpeg，跳过生成 MP4:", out_mp4)
        return
    frame_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(frame_dir / "frame_%06d.png")
    cmd = [
        ff,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-framerate",
        str(fps),
        "-i",
        pattern,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(out_mp4),
    ]
    subprocess.run(cmd, check=True)
    print(f"[mp4] 已写入 {out_mp4}")


@torch.no_grad()
def evaluate_scene(
    scene_name: str,
    dataset_path: Path,
    dataset_config: Dict[str, Any],
    model: DepthModule,
    viz: DepthVizCallback,
    depth_evaluator: DepthEvaluator,
    device: torch.device,
    use_amp: bool,
    fps: float,
    test_viz_root: Path,
) -> int:
    """对单场景逐帧推理；返回写入的帧数。"""
    scene_path = dataset_path / str(scene_name)
    assert scene_path.is_dir(), f"场景目录不存在: {scene_path}"

    rnn_states = RNNStates()
    worker_id = 0
    frame_idx = 0

    seq_dir = test_viz_root / f"seq_{scene_name}"
    dir_pred_full = seq_dir / "pred_full_frames"
    dir_masked = seq_dir / "masked_frames"
    dir_pred_full.mkdir(parents=True, exist_ok=True)
    dir_masked.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt

    it = iter_liosam_scene_frames(scene_path, dataset_config)
    for is_first, payload in it:
        sample = dict(payload)
        sample["is_first"] = is_first
        batch = _collate_like_streaming(sample)

        data = batch["data"]
        is_first_sample = data[DataType.IS_FIRST_SAMPLE]
        rnn_states.reset(worker_id=worker_id, indices_or_bool_tensor=is_first_sample)
        prev_states = rnn_states.get_states(worker_id=worker_id)

        # 与 training_step / DepthVizCallback 一致：可视化深度应对齐「未 pad」的事件分辨率 input_hw，
        # 而非模型内部的 348 宽 padding 网格（右侧 pad 列常为 0 → colormap 视为无效变黑，观感像「细节全无」）。
        ev_tensor_orig = data[DataType.EV_REPR][0]
        input_hw = tuple(int(x) for x in ev_tensor_orig.shape[-2:])

        ev_tensor = ev_tensor_orig.to(device=device)
        depth_gt_log = data[DataType.DEPTH][0].to(device=device, dtype=torch.float32)
        depth_mask = data[DataType.DEPTH_MASK][0].to(device=device)

        ev_tensor = model.input_padder.pad_tensor_ev_repr(ev_tensor.unsqueeze(0))
        depth_gt = model.input_padder.pad_tensor_ev_repr(depth_gt_log.unsqueeze(0))
        depth_gt_norm_log = model.log_depth_to_norm_log_depth(depth_gt)

        if depth_mask.dim() == 2:
            depth_mask = depth_mask.unsqueeze(0)
        depth_mask_padded = model.input_padder.pad_tensor_ev_repr(depth_mask.unsqueeze(0))

        with torch.cuda.amp.autocast(enabled=use_amp and device.type == "cuda"):
            predictions, _, prev_states = model.mdl(
                x=ev_tensor,
                previous_states=prev_states,
                retrieve_depth=True,
                targets=None,
                masks=None,
            )

        rnn_states.save_states_and_detach(worker_id=worker_id, states=prev_states)

        depth_pred = _finest_depth_pred(predictions)
        pred_hw = depth_pred.shape[-2:]
        depth_gt_m = depth_gt_norm_log
        if depth_gt_m.shape[-2:] != pred_hw:
            depth_gt_m = F.interpolate(depth_gt_m, size=pred_hw, mode="bilinear", align_corners=False)
        mask_m = depth_mask_padded
        if mask_m is not None and mask_m.shape[-2:] != pred_hw:
            mask_m = F.interpolate(mask_m.float(), size=pred_hw, mode="nearest").bool()

        depth_evaluator.add_predictions(depth_pred, depth_gt_m, mask_m)

        # 与 training_step 中 depth_pred_for_viz：先对齐到原始事件分辨率再转 meters（见 depth_viz.on_train_batch_end）
        pred_norm = depth_pred[0:1]
        pred_for_viz = F.interpolate(
            pred_norm,
            size=input_hw,
            mode="bilinear",
            align_corners=False,
        )
        pred_real = viz._norm_log_to_depth_real(pred_for_viz).squeeze().detach().cpu().numpy()

        # 与 DepthVizCallback：若仍有宽度不一致（极少见），裁到与事件宽一致，去掉右侧 pad 伪影
        _, W_d = pred_real.shape[-2:]
        _, W_ev = input_hw
        if W_d != W_ev and pred_real.ndim >= 2:
            pred_real = pred_real[..., :W_ev]

        if mask_m is not None:
            mask_for_viz = F.interpolate(
                mask_m.float(),
                size=input_hw,
                mode="nearest",
            ).bool()
            mask_hw = mask_for_viz.squeeze(0).squeeze(0).detach().cpu().numpy()
            if mask_hw.shape[-1] != pred_real.shape[-1]:
                mask_hw = mask_hw[..., : pred_real.shape[-1]]
        else:
            mask_hw = None

        img_full = viz._depth_to_colormap(
            pred_real,
            vmin=viz.depth_min,
            vmax=viz.depth_max,
            mask=None,
            is_error=False,
        )
        img_masked = viz._depth_to_colormap(
            pred_real,
            vmin=viz.depth_min,
            vmax=viz.depth_max,
            mask=mask_hw,
            is_error=False,
        )

        pf = dir_pred_full / f"frame_{frame_idx:06d}.png"
        pm = dir_masked / f"frame_{frame_idx:06d}.png"
        plt.imsave(str(pf), img_full)
        plt.imsave(str(pm), img_masked)

        frame_idx += 1

    # MP4
    out_full = test_viz_root / f"seq_{scene_name}_pred_full.mp4"
    out_masked = test_viz_root / f"seq_{scene_name}_masked.mp4"
    images_to_mp4(dir_pred_full, out_full, fps=fps)
    images_to_mp4(dir_masked, out_masked, fps=fps)

    return frame_idx


def main() -> None:
    parser = argparse.ArgumentParser(description="LiOSAM 测试集评估（序列 22、23）")
    parser.add_argument("--config", type=str, default="config_liosam.yaml")
    parser.add_argument("--ckpt", type=str, required=True, help="Lightning 或纯 state_dict 的 .ckpt / .pt")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--scenes",
        type=str,
        default="22,23",
        help="逗号分隔的场景编号，默认 22,23",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="视频帧率，默认由 frame_interval_sec 推导（1/间隔，例如 5ms -> 200）",
    )
    args = parser.parse_args()

    import yaml

    config_path = Path(args.config)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    dataset_config = config["dataset"]
    dataset_path = Path(dataset_config["path"])
    scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]

    fi = float(dataset_config.get("frame_interval_sec", 0.005))
    fps = args.fps if args.fps is not None else round(1.0 / fi)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}, fps={fps}（约每 {1000.0/fps:.3f} ms 一帧）")

    precision = config.get("training", {}).get("precision", "32")
    use_amp = precision == "16-mixed"

    model = DepthModule(config)
    load_model_weights(model, Path(args.ckpt))
    model = model.to(device)
    model.eval()

    depth_cfg = config.get("model", {}).get("depth_range", {})
    min_d = float(depth_cfg.get("min", 0.5))
    max_d = float(depth_cfg.get("max", 80.0))
    depth_evaluator = DepthEvaluator(min_depth=min_d, max_depth=max_d)

    viz = DepthVizCallback(config)

    test_viz_root = Path("test_viz")
    test_viz_root.mkdir(parents=True, exist_ok=True)

    total_frames = 0
    for scene in scenes:
        n = evaluate_scene(
            scene_name=scene,
            dataset_path=dataset_path,
            dataset_config=dataset_config,
            model=model,
            viz=viz,
            depth_evaluator=depth_evaluator,
            device=device,
            use_amp=use_amp,
            fps=float(fps),
            test_viz_root=test_viz_root,
        )
        total_frames += n
        print(f"[scene {scene}] 帧数={n}")

    if total_frames == 0 or not depth_evaluator.has_data():
        print("[error] 未加载任何帧，请检查场景目录与 index.txt。")
        sys.exit(1)

    metrics = depth_evaluator.evaluate_buffer()
    d1 = metrics.get("delta1", float("nan"))
    rmse = metrics.get("rmse", float("nan"))
    print(f"[test set 全体 {total_frames} 帧] delta1={d1:.6f}, rmse={rmse:.6f}")

    write_local_loss_metrics(delta1=d1, rmse=rmse)
    print("[done] 可视化目录:", test_viz_root.resolve())


if __name__ == "__main__":
    main()

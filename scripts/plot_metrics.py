#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys

import matplotlib.pyplot as plt
import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
LOCAL_DIR = os.path.join(ROOT, "local_loss")
OUT_DIR = os.path.join(ROOT, "local_loss")


def _load_xy(path):
    if not os.path.isfile(path):
        return None, None
    steps = []
    vals = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                step_str, val_str = line.split()[:2]
                steps.append(int(step_str))
                vals.append(float(val_str))
            except ValueError:
                continue
    if not steps:
        return None, None
    return np.array(steps), np.array(vals)


def load_train_loss():
    return _load_xy(os.path.join(LOCAL_DIR, "train_loss.txt"))


def load_train_delta1():
    return _load_xy(os.path.join(LOCAL_DIR, "train_delta1.txt"))


def load_val_loss():
    return _load_xy(os.path.join(LOCAL_DIR, "val_rmse.txt"))


def load_val_delta1():
    return _load_xy(os.path.join(LOCAL_DIR, "val_delta1.txt"))


def load_train_lr():
    return _load_xy(os.path.join(LOCAL_DIR, "train_lr.txt"))


def _plot_curve(steps, vals, ylabel, title, filename):
    if steps is None or vals is None:
        return
    if not os.path.isdir(OUT_DIR):
        os.makedirs(OUT_DIR, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.plot(steps, vals, linewidth=1.0)
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.4)
    out_path = os.path.join(OUT_DIR, filename)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_all():
    # 1) 训练 loss
    s, v = load_train_loss()
    _plot_curve(s, v, "loss", "Train loss vs. epoch", "train_loss_curve.png")
    # 2) 训练 delta1
    s, v = load_train_delta1()
    _plot_curve(s, v, "delta1", "Train δ1 vs. epoch", "train_delta1_curve.png")
    # 3) 验证 loss（这里用 rmse 作为验证 loss）
    s, v = load_val_loss()
    _plot_curve(s, v, "rmse", "Val loss (RMSE) vs. epoch", "val_loss_curve.png")
    # 4) 验证 delta1
    s, v = load_val_delta1()
    _plot_curve(s, v, "delta1", "Val δ1 vs. epoch", "val_delta1_curve.png")
    # 5) 学习率
    s, v = load_train_lr()
    _plot_curve(s, v, "learning rate", "Learning rate vs. epoch", "train_lr_curve.png")


def main():
    # 从 local_loss 中的四个日志文件绘制四条曲线。
    plot_all()


if __name__ == "__main__":
    main()


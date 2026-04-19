#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的训练脚本 - 使用单一YAML配置文件

用法:
    python train.py                          # 使用默认config.yaml (Virtual数据集)
    python train.py --config config_dsec.yaml  # 使用DSEC配置
    python train.py --config config.yaml --gpus 0,1  # 命令行覆盖
"""
 
import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import torch

import pytorch_lightning as pl
import yaml
from typing import Dict, Any
from pytorch_lightning.callbacks import ModelCheckpoint

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from callbacks.custom import get_ckpt_callback, get_viz_callback
from callbacks.gradflow import GradFlowLogCallback
from loggers.wandb_logger import WandbLogger
from modules.data.event_data_module import DataModule
from modules.depth_estimation import Module as DepthModule


def load_config(config_path: str) -> Dict[str, Any]:
    """加载YAML配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    return config_dict


def override_config(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """使用命令行参数覆盖配置"""
    if args.gpus is not None:
        if ',' in args.gpus:
            config['hardware']['gpus'] = [int(g) for g in args.gpus.split(',')]
        else:
            config['hardware']['gpus'] = int(args.gpus)
    
    # 仅当命令行显式传入 --batch_size 时覆盖配置文件
    if args.batch_size is not None:
        config['batch_size']['train'] = args.batch_size
        config['batch_size']['eval'] = args.batch_size
    
    if args.lr is not None:
        config['training']['learning_rate'] = args.lr
    
    if args.max_epochs is not None:
        config['training']['max_epochs'] = args.max_epochs
    
    # 随机种子：命令行优先，其次是配置文件原有值
    if args.seed is not None:
        config['reproduce']['seed_everything'] = args.seed
    
    # 单 batch 调试：如果显式指定了起始 npz 下标，则写入 dataset 配置
    # 实际使用逻辑在 data/genx_utils/dataset_rnd.py 里，仅对 LiOSAM + RANDOM 模式的 train 有效
    if getattr(args, "debug_start_npz_idx", None) is not None:
        if 'dataset' not in config:
            config['dataset'] = {}
        config['dataset']['debug_start_npz_index'] = int(args.debug_start_npz_idx)
    
    return config


def setup_wandb_logger(config: Dict[str, Any]) -> Optional[WandbLogger]:
    """设置WandB日志"""
    if config.get('wandb', None) is None:
        return None
    
    # log_model=False：checkpoint 仅本地保存，不上传至 wandb 云端；其他指标/图像等照常记录
    logger = WandbLogger(
        project=config['wandb']['project_name'],
        name=config['wandb']['group_name'],
        config=config,
        log_model=False,
    )
    
    return logger


def main():
    parser = argparse.ArgumentParser(description='训练深度估计模型')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='配置文件路径 (默认: config.yaml)')
    parser.add_argument('--gpus', type=str, default='0',
                        help='GPU设置 (例如: "0" 或 "0,1,2,3")')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size（不传则使用配置文件中的值）')
    parser.add_argument('--lr', type=float, default=0.002,
                        help='学习率')
    parser.add_argument('--max_epochs', type=int, default=50,
                        help='最大训练轮数')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子（不填则使用配置文件中的 reproduce.seed_everything）')
    parser.add_argument('--debug_fixed_batch', action='store_true',
                        help='调试模式：在单个 batch 上反复训练，检查 loss 是否能下降')
    parser.add_argument('--debug_start_npz_idx', type=int, default=None,
                        help='单 batch 调试时：指定 batch 对应的 LiOSAM 序列起始于第几个 npz（index.txt 中的行号，从 0 开始）')
    
    args = parser.parse_args()
    
    # 加载配置
    print(f"[CONFIG] 加载配置文件: {args.config}")
    config = load_config(args.config)
    
    # 命令行覆盖（包括随机种子）
    config = override_config(config, args)
    
    # 设置随机种子（优先使用命令行覆盖后的 config['reproduce']['seed_everything']）
    seed_value = config.get('reproduce', {}).get('seed_everything')
    if seed_value is not None:
        pl.seed_everything(seed_value)
        print(f"[SEED] 设置随机种子: {seed_value}")
    else:
        print("[SEED] 未设置随机种子（结果可能不可复现）")
    
    # 打印关键配置
    print("\n" + "="*70)
    print("训练配置")
    print("="*70)
    print(f"数据集: {config['dataset']['name']}")
    print(f"数据路径: {config['dataset']['path']}")
    print(f"模型: {config['model']['name']} ({config['model']['backbone']['name']})")
    print(f"Batch Size: {config['batch_size']['train']}")
    print(f"学习率: {config['training']['learning_rate']}")
    print(f"最大Epochs: {config['training']['max_epochs']}")
    print(f"GPU: {config['hardware']['gpus']}")
    print(f"随机种子: {config.get('reproduce', {}).get('seed_everything', '未设置')}")
    print(f"WandB项目: {config['wandb']['project_name']} / {config['wandb']['group_name']}")
    print("="*70 + "\n")
    
    # 创建数据模块
    print("[DATA] 创建数据模块...")
    data_module = DataModule(
        dataset_config=config['dataset'],
        num_workers_train=config['hardware']['num_workers']['train'],
        num_workers_eval=config['hardware']['num_workers']['eval'],
        batch_size_train=config['batch_size']['train'],
        batch_size_eval=config['batch_size']['eval']
    )
    
    # 创建模型模块
    print("[MODEL] 创建模型...")
    model = DepthModule(config)
    model.save_debug_depth = True
    
    # 设置WandB日志
    wandb_logger = setup_wandb_logger(config)
    
    # 创建callbacks
    callbacks = []
    
    # Checkpoint callback
    ckpt_callback = get_ckpt_callback(config)
    callbacks.append(ckpt_callback)
    
    # 可视化callback
    viz_callback = get_viz_callback(config)
    callbacks.append(viz_callback)
    
    # 梯度流callback
    gradflow_callback = GradFlowLogCallback(
        log_every_n_train_steps=config['logging']['train']['log_every_n_steps']
    )
    callbacks.append(gradflow_callback)
    
    # 配置Trainer
    trainer_args = {
        'max_epochs': config['training']['max_epochs'],
        'max_steps': config['training'].get('max_steps'),
        'precision': config['training']['precision'],
        'gradient_clip_val': config['training']['gradient_clip_val'],
        'limit_train_batches': config['training']['limit_train_batches'],
        'limit_val_batches': config['validation']['limit_val_batches'],
        'val_check_interval': config['validation']['val_check_interval'],
        'check_val_every_n_epoch': config['validation'].get('check_val_every_n_epoch'),
        'log_every_n_steps': config['logging']['train']['log_every_n_steps'],
        'callbacks': callbacks,
        'logger': wandb_logger,
        'deterministic': config['reproduce']['deterministic_flag'],
        'benchmark': config['reproduce']['benchmark'],
    }

    # 固定单个 batch 过拟合调试模式
    if args.debug_fixed_batch:
        print("\n[DEBUG] 启用固定单个 batch 过拟合调试模式")
        print("       Trainer 将只在 1 个训练 batch 上反复优化，用于检查 loss 是否下降。")
        # 只使用 1 个训练 batch（Lightning 会反复用这一批做前向 + 反向）
        trainer_args['overfit_batches'] = 1
        # 调试时可以关掉验证，加快迭代，专注看训练 loss
        trainer_args['limit_val_batches'] = 0
        trainer_args['check_val_every_n_epoch'] = None
    
    # GPU设置（自动检测 CUDA 可用性）
    # import torch
    cuda_available = torch.cuda.is_available()
    
    if cuda_available:
        print("[CUDA] CUDA 可用，使用 GPU 训练")
        gpus = config['hardware']['gpus']
        if isinstance(gpus, int):
            if gpus == -1:
                trainer_args['accelerator'] = 'gpu'
                trainer_args['devices'] = 'auto'
                num_gpus = torch.cuda.device_count()
                if num_gpus > 1:
                    trainer_args['strategy'] = 'ddp'
                print(f"[CUDA] 使用全部 {num_gpus} 张 GPU")
            elif gpus >= 0:
                trainer_args['accelerator'] = 'gpu'
                trainer_args['devices'] = [gpus]
        elif isinstance(gpus, list):
            trainer_args['accelerator'] = 'gpu'
            trainer_args['devices'] = gpus
            if len(gpus) > 1:
                trainer_args['strategy'] = 'ddp'
    else:
        print("[WARNING] CUDA 不可用，使用 CPU 训练")
        trainer_args['accelerator'] = 'cpu'
        trainer_args['devices'] = 1
    
    print("[TRAINER] 创建Trainer...")
    trainer = pl.Trainer(**trainer_args)
    
    # 开始训练
    print("\n" + "="*70)
    print("开始训练...")
    print("="*70 + "\n")
    
    try:
        trainer.fit(model, datamodule=data_module)
        print("\n[SUCCESS] 训练完成!")
    except KeyboardInterrupt:
        print("\n[WARNING] 训练被中断")
    except Exception as e:
        print(f"\n[ERROR] 训练出错: {e}")
        raise


if __name__ == '__main__':
    main()


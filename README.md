# Event-based Depth Estimation with MobileNetV2 + RNN

<p align="center">
  <strong>深度估计系统 - 基于事件相机的深度预测</strong>
</p>

本项目实现了基于事件相机的深度估计系统，使用轻量级的MobileNetV2 + RNN backbone和UNet风格的深度估计head。

**特点:**
- ✅ 轻量级MobileNetV2 + RNN backbone
- ✅ UNet风格的多尺度深度解码器
- ✅ 组合损失函数（BerHu + SILog + Gradient）
- ✅ Log域深度表示
- ✅ 完整的训练和评估流程
- ✅ Virtual数据集生成工具

## 快速开始

### 1. 安装

#### 使用Conda（推荐）
```bash
conda create -y -n evdepth python=3.9 pip
conda activate evdepth
conda config --set channel_priority flexible

# 安装PyTorch（根据您的CUDA版本调整）
CUDA_VERSION=11.8
conda install -y pytorch=2.0.0 torchvision=0.15.0 pytorch-cuda=$CUDA_VERSION \
    -c pytorch -c nvidia

# 安装其他依赖
conda install -y h5py=3.8.0 blosc-hdf5-plugin=1.0.0 \
    hydra-core=1.3.2 einops=0.6.0 tqdm numba \
    -c conda-forge

pip install -r requirements.txt
```

#### 使用pip
```bash
# 安装PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 安装项目依赖
pip install -r requirements.txt
```

### 2. 生成Virtual数据集（测试用）

```bash
python scripts/generate_virtual_depth_data.py \
    --output_dir data/virtual_depth \
    --num_train 3 \
    --num_val 1 \
    --num_test 1 \
    --num_frames 100
```

### 3. 测试安装

```bash
# 快速测试所有组件
python test_depth_setup.py

# 端到端测试（包含数据生成和训练）
python test_end_to_end.py
```

### 4. 开始训练

```bash
# 最简单的方式 - 使用默认配置
python train.py

# 指定GPU和batch size
python train.py --gpus 0 --batch_size 8

# 使用DSEC数据集
python train.py --config config_dsec.yaml --data_path /path/to/dsec
```

**详细说明**: 查看 `SIMPLE_CONFIG_GUIDE.md`

## 支持的数据集

### Virtual Dataset（虚拟数据集）
- **用途**: 快速测试和验证代码
- **特点**: 自动生成，包含多种场景类型
- **生成**: `scripts/generate_virtual_depth_data.py`

### DSEC Dataset
- **全称**: Driving Stereo Event Camera Dataset
- **网站**: [DSEC](https://dsec.ifi.uzh.ch/)
- **特点**: 真实驾驶场景，包含单目事件相机和深度真值
- **配置**: `config_dsec.yaml`

## 项目结构

```
evdepth/RVT_TB/
├── models/                      # 模型实现
│   ├── depth/                   # 深度估计模型
│   │   ├── depth_head.py        # UNet风格解码器
│   │   ├── depth_detector.py    # 主模型
│   │   └── depth_losses.py      # 损失函数
│   ├── detection/
│   │   ├── recurrent_backbone/  # Backbone实现
│   │   │   └── mobilenet_rnn.py # MobileNetV2 + RNN
│   │   └── yolox_extension/
│   │       └── models/
│   │           ├── build.py     # 构建函数
│   │           └── yolo_pafpn.py # FPN
│   └── layers/
│       └── rnn.py               # LSTM层
│
├── modules/                     # PyTorch Lightning模块
│   ├── depth_estimation.py      # 深度估计训练模块
│   ├── data/
│   │   └── genx.py              # 数据模块
│   └── utils/
│       ├── detection.py         # 辅助工具
│       └── fetch.py             # 模块加载
│
├── data/                        # 数据处理
│   ├── genx_utils/              # 数据加载工具
│   │   ├── depth_loader.py      # 深度数据加载
│   │   ├── sequence_base.py     # 序列基类
│   │   ├── sequence_rnd.py      # 随机访问
│   │   └── sequence_for_streaming.py  # 流式加载
│   └── utils/
│       ├── types.py             # 类型定义
│       ├── spatial.py           # 空间工具
│       └── augmentor.py         # 数据增强
│
├── callbacks/                   # 训练回调
│   ├── depth_viz.py             # 深度可视化
│   ├── gradflow.py              # 梯度流监控
│   ├── viz_base.py              # 可视化基类
│   └── custom.py                # 工厂函数
│
├── utils/                       # 工具函数
│   ├── evaluation/
│   │   └── depth/               # 深度评估
│   │       └── depth_metrics.py # 评估指标
│   ├── padding.py
│   ├── timers.py
│   └── helpers.py
│
├── scripts/                     # 工具脚本
│   └── generate_virtual_depth_data.py  # 数据生成
│
├── loggers/                     # 日志记录
│   ├── wandb_logger.py
│   └── utils.py
│
├── config.yaml                  # ⭐ Virtual数据集配置
├── config_dsec.yaml             # ⭐ DSEC数据集配置
├── train.py                     # ⭐ 训练脚本
├── test_depth_setup.py          # 组件测试
├── test_end_to_end.py           # 端到端测试
│
└── 文档/
    ├── README.md                # 本文档
    ├── SIMPLE_CONFIG_GUIDE.md   # ⭐ 配置指南
    ├── QUICK_START.md           # 快速开始
    ├── DEPTH_ESTIMATION_README.md  # 完整技术文档
    ├── PROJECT_STRUCTURE.md     # 项目结构详解
    └── ...                      # 其他文档
```

## 模型架构

```
事件输入 (B, 20, H, W)
    ↓
MobileNetV2 + RNN Backbone
    ├── Stage 1: /4  → 64 channels  + LSTM
    ├── Stage 2: /8  → 128 channels + LSTM
    ├── Stage 3: /16 → 256 channels + LSTM
    └── Stage 4: /32 → 512 channels + LSTM
    ↓
FPN (Feature Pyramid Network)
    ├── /8  - 256 channels
    ├── /16 - 512 channels
    └── /32 - 1024 channels
    ↓
UNet-style Depth Decoder
    ├── /32 → /16 (上采样 + 跳连)
    ├── /16 → /8  (上采样 + 跳连)
    ├── /8  → /4  (上采样)
    └── /4  → /2  (上采样)
    ↓
多尺度深度输出 (Log空间)
    ├── depth_16 (1/16分辨率)
    ├── depth_8  (1/8分辨率)
    ├── depth_4  (1/4分辨率)
    └── depth_2  (1/2分辨率，最精细)
```

## 配置系统

本项目使用**简单直观的单文件配置系统**：

**配置文件**:
- `config.yaml` - Virtual数据集配置（默认）
- `config_dsec.yaml` - DSEC数据集配置

**使用方式**:
```bash
# 使用默认配置
python train.py

# 命令行覆盖
python train.py --gpus 0 --batch_size 8 --lr 0.0001

# 使用DSEC配置
python train.py --config config_dsec.yaml
```

**详细文档**: `SIMPLE_CONFIG_GUIDE.md` ⭐

---

## 训练示例

### Virtual数据集（快速测试）

```bash
# 生成数据
python scripts/generate_virtual_depth_data.py \
    --output_dir data/virtual_depth

# 训练（最简单）
python train.py

# 或指定参数
python train.py --data_path data/virtual_depth --gpus 0 --batch_size 4
```

### DSEC数据集（真实数据）

```bash
python train.py --config config_dsec.yaml --data_path /path/to/dsec --gpus 0,1
```

## 评估指标

- **RMSE** (Root Mean Square Error)
- **MAE** (Mean Absolute Error)
- **AbsRel** (Absolute Relative Error)
- **SqRel** (Squared Relative Error)
- **δ < 1.25, 1.25², 1.25³** (阈值准确率)

## 损失函数

- **BerHu Loss**: 鲁棒的深度损失
- **SILog Loss**: 尺度不变对数损失
- **Gradient Loss**: 空间梯度平滑损失
- **多尺度**: 4个分辨率的组合损失

## 配置修改

### 快速修改（简化配置）

直接编辑 `config.yaml` 或 `config_dsec.yaml`：

```yaml
# 修改数据集
dataset:
  path: data/my_dataset
  sequence_length: 5

# 修改模型
model:
  backbone:
    embed_dim: 128  # 增大模型
  loss:
    berhu_weight: 1.0
    silog_weight: 1.0

# 修改训练
training:
  learning_rate: 0.0001
  max_epochs: 200

batch_size:
  train: 16
```

或使用命令行：

```bash
python train.py --batch_size 16 --lr 0.0001 --max_epochs 200
```

### 详细配置说明

查看配置文件中的注释，所有参数都有详细说明。

**完整文档**: `SIMPLE_CONFIG_GUIDE.md`

## 文档

### 快速开始
- **[SIMPLE_CONFIG_GUIDE.md](SIMPLE_CONFIG_GUIDE.md)** ⭐ - 配置系统使用指南（推荐）
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - 快速参考卡片
- **[QUICK_START.md](QUICK_START.md)** - 3步快速开始

### 技术文档
- **[DEPTH_ESTIMATION_README.md](DEPTH_ESTIMATION_README.md)** - 完整技术文档
- **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)** - 项目结构详解
- **[BACKBONE_SIMPLIFIED.md](BACKBONE_SIMPLIFIED.md)** - Backbone架构

### 参考文档
- **[MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)** - 配置系统迁移指南（如果您之前使用过旧版本）
- **[FINAL_SETUP_GUIDE.md](FINAL_SETUP_GUIDE.md)** - 设置指南
- **[CALLBACKS_SUMMARY.md](CALLBACKS_SUMMARY.md)** - Callbacks说明
- **[PROJECT_CLEANUP_SUMMARY.md](PROJECT_CLEANUP_SUMMARY.md)** - 项目清理总结

## 常见问题

**Q: 配置文件在哪里？**
A: 项目根目录的 `config.yaml`（Virtual数据集）和 `config_dsec.yaml`（DSEC数据集）。

**Q: 如何修改配置？**
A: 直接编辑YAML文件或使用命令行参数（如 `--batch_size 8`）。详见 `SIMPLE_CONFIG_GUIDE.md`。

**Q: Virtual数据集和真实数据集的区别？**
A: Virtual是合成数据，用于快速测试代码。真实数据集（如DSEC）用于实际性能评估。

**Q: 如何准备DSEC数据？**
A: 参考`DEPTH_ESTIMATION_README.md`中的数据格式说明。

**Q: 训练需要多少GPU内存？**
A: batch_size=8时约需要8-12GB。可以调整batch size适配您的GPU。

**Q: 如何调整模型大小？**
A: 在配置文件中修改`model.backbone.embed_dim`和`num_blocks`参数。详见`BACKBONE_SIMPLIFIED.md`。

## 可视化文件说明
check_input 检查输入数据，取epoch 0的所有batch，batch里每个样本都展示，跨batch连续计数，长序列只取每个序列的最后一帧
depth_epoch_viz_train 训练阶段，每个epoch只对第一个batch的结果可视化
depth_epoch_viz 验证阶段，对收集的多个batch的
## 引用

本项目基于RVT（Recurrent Vision Transformers）架构，但已完全改造为深度估计系统。

```bibtex
@InProceedings{Gehrig_2023_CVPR,
  author  = {Mathias Gehrig and Davide Scaramuzza},
  title   = {Recurrent Vision Transformers for Object Detection with Event Cameras},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year    = {2023},
}
```

## 许可

遵循原RVT项目的许可协议。

## 相关资源

- [DSEC Dataset](https://dsec.ifi.uzh.ch/) - 驾驶场景事件相机数据集
- [MVSEC Dataset](https://daniilidis-group.github.io/mvsec/) - 多模态事件相机数据集
- [E2Depth](https://github.com/uzh-rpg/rpg_e2depth) - 事件相机深度估计

---

**项目类型**: 深度估计  
**Backbone**: MobileNetV2 + RNN  
**支持数据集**: Virtual, DSEC  
**最后更新**: 2025-12-01

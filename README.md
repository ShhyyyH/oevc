# 图像识别学习项目

图像识别和数据分析 —— 基于 PyTorch 的卷积神经网络（CNN）图像分类

## 项目结构

```
├── model.py          # CNN 模型定义（CIFAR-10 / MNIST）
├── train.py          # 训练脚本（记录损失、准确率，保存检查点）
├── predict.py        # 预测脚本（加载模型，可视化预测结果）
├── utils.py          # 工具函数（指标记录、学习曲线绘制）
├── requirements.txt  # Python 依赖
├── data/             # 数据集（自动下载）
├── checkpoints/      # 模型检查点及训练历史
└── results/          # 预测结果可视化
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 训练模型

训练 CIFAR-10（默认）：

```bash
python train.py --dataset cifar10 --epochs 50
```

训练 MNIST：

```bash
python train.py --dataset mnist --epochs 20
```

更多参数：

```bash
python train.py --help
```

### 3. 预测

从测试集随机抽取样本并可视化：

```bash
python predict.py --dataset cifar10 --n-samples 16
```

对单张图像预测：

```bash
python predict.py --image path/to/image.jpg --dataset cifar10
```

## 训练过程记录

训练过程中会自动保存：

- `checkpoints/training_history.json` — 每轮的损失和准确率
- `checkpoints/best_model.pth` — 验证集上表现最好的模型
- `checkpoints/checkpoint_epoch_N.pth` — 每 10 轮保存一次检查点
- `checkpoints/learning_curves.png` — 学习曲线图（损失 & 准确率）

## 模型说明

### CIFAR-10 CNN（`ImageRecognitionCNN`）

- 输入：32×32 彩色图像（3 通道）
- 结构：3 个卷积块（含 BatchNorm + Dropout）+ 全连接层
- 分类数：10

### MNIST CNN（`SimpleCNN`）

- 输入：28×28 灰度图像（1 通道）
- 结构：2 个卷积块 + 全连接层
- 分类数：10

## 依赖环境

- Python 3.8+
- PyTorch 1.12+
- torchvision 0.13+

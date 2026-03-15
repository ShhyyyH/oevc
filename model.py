"""
卷积神经网络模型定义
图像识别 CNN 模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """卷积块：卷积 + 批归一化 + 激活函数"""

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ImageRecognitionCNN(nn.Module):
    """
    用于图像识别的卷积神经网络
    适用于 CIFAR-10 数据集（32x32 彩色图像，10 类）
    """

    def __init__(self, num_classes=10, in_channels=3):
        super(ImageRecognitionCNN, self).__init__()

        # 特征提取层
        self.features = nn.Sequential(
            # 第一卷积块: 3 -> 32 通道
            ConvBlock(in_channels, 32),
            ConvBlock(32, 32),
            nn.MaxPool2d(2, 2),   # 32x32 -> 16x16
            nn.Dropout2d(0.25),

            # 第二卷积块: 32 -> 64 通道
            ConvBlock(32, 64),
            ConvBlock(64, 64),
            nn.MaxPool2d(2, 2),   # 16x16 -> 8x8
            nn.Dropout2d(0.25),

            # 第三卷积块: 64 -> 128 通道
            ConvBlock(64, 128),
            ConvBlock(128, 128),
            nn.MaxPool2d(2, 2),   # 8x8 -> 4x4
            nn.Dropout2d(0.25),
        )

        # 分类层
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class SimpleCNN(nn.Module):
    """
    简单版卷积神经网络（用于快速实验）
    适用于 MNIST 数据集（28x28 灰度图像，10 类）
    """

    def __init__(self, num_classes=10, in_channels=1):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),   # 28x28 -> 14x14

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),   # 14x14 -> 7x7
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def get_model(model_name='cifar', num_classes=10):
    """
    获取模型

    参数:
        model_name: 模型名称，'cifar' 或 'mnist'
        num_classes: 分类数量

    返回:
        模型实例
    """
    if model_name == 'cifar':
        return ImageRecognitionCNN(num_classes=num_classes, in_channels=3)
    elif model_name == 'mnist':
        return SimpleCNN(num_classes=num_classes, in_channels=1)
    else:
        raise ValueError(f"未知模型名称: {model_name}，支持 'cifar' 或 'mnist'")


if __name__ == '__main__':
    # 测试模型结构
    print("=== CIFAR-10 CNN 模型 ===")
    model_cifar = get_model('cifar', num_classes=10)
    x = torch.randn(4, 3, 32, 32)
    out = model_cifar(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {out.shape}")
    total_params = sum(p.numel() for p in model_cifar.parameters())
    print(f"参数总量: {total_params:,}")

    print("\n=== MNIST CNN 模型 ===")
    model_mnist = get_model('mnist', num_classes=10)
    x = torch.randn(4, 1, 28, 28)
    out = model_mnist(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {out.shape}")
    total_params = sum(p.numel() for p in model_mnist.parameters())
    print(f"参数总量: {total_params:,}")

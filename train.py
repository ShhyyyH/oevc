"""
图像识别模型训练脚本
记录并保存训练过程中的损失和准确率
"""

import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision
import torchvision.transforms as transforms

from model import get_model
from utils import (
    plot_training_history,
    save_training_history,
    AverageMeter,
    compute_accuracy,
)


# ========== 数据集配置 ==========
DATASET_CONFIGS = {
    'cifar10': {
        'model_name': 'cifar',
        'num_classes': 10,
        'mean': (0.4914, 0.4822, 0.4465),
        'std': (0.2470, 0.2435, 0.2616),
        'classes': ['飞机', '汽车', '鸟', '猫', '鹿',
                    '狗', '青蛙', '马', '船', '卡车'],
    },
    'mnist': {
        'model_name': 'mnist',
        'num_classes': 10,
        'mean': (0.1307,),
        'std': (0.3081,),
        'classes': [str(i) for i in range(10)],
    },
}


def get_data_loaders(dataset_name, data_dir='./data', batch_size=128, num_workers=2):
    """
    加载并预处理数据集

    参数:
        dataset_name: 数据集名称，'cifar10' 或 'mnist'
        data_dir: 数据存储目录
        batch_size: 批次大小
        num_workers: 数据加载线程数

    返回:
        train_loader, test_loader
    """
    config = DATASET_CONFIGS[dataset_name]

    if dataset_name == 'cifar10':
        # CIFAR-10 数据增强
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(config['mean'], config['std']),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(config['mean'], config['std']),
        ])
        train_dataset = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=train_transform
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=test_transform
        )

    elif dataset_name == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(config['mean'], config['std']),
        ])
        train_dataset = torchvision.datasets.MNIST(
            root=data_dir, train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.MNIST(
            root=data_dir, train=False, download=True, transform=transform
        )
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    return train_loader, test_loader


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    """
    训练一个 epoch

    返回:
        (平均损失, 准确率)
    """
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        acc = compute_accuracy(outputs, labels)
        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(acc, images.size(0))

        if (batch_idx + 1) % 50 == 0:
            print(f"  Epoch [{epoch}] Step [{batch_idx + 1}/{len(loader)}] "
                  f"Loss: {loss_meter.avg:.4f}  Acc: {acc_meter.avg:.2f}%")

    return loss_meter.avg, acc_meter.avg


def evaluate(model, loader, criterion, device):
    """
    在验证集/测试集上评估模型

    返回:
        (平均损失, 准确率)
    """
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            acc = compute_accuracy(outputs, labels)
            loss_meter.update(loss.item(), images.size(0))
            acc_meter.update(acc, images.size(0))

    return loss_meter.avg, acc_meter.avg


def train(
    dataset_name='cifar10',
    epochs=50,
    batch_size=128,
    learning_rate=0.001,
    weight_decay=1e-4,
    data_dir='./data',
    save_dir='./checkpoints',
    num_workers=2,
):
    """
    完整训练流程

    参数:
        dataset_name: 数据集名称
        epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 初始学习率
        weight_decay: 权重衰减（L2 正则化）
        data_dir: 数据目录
        save_dir: 模型保存目录
        num_workers: 数据加载线程数
    """
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 数据加载
    print(f"\n加载 {dataset_name} 数据集...")
    train_loader, test_loader = get_data_loaders(
        dataset_name, data_dir, batch_size, num_workers
    )

    # 模型、损失函数、优化器
    config = DATASET_CONFIGS[dataset_name]
    model = get_model(config['model_name'], num_classes=config['num_classes'])
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # 学习历史记录
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': [],
        'epoch_time': [],
    }

    best_val_acc = 0.0
    print(f"\n开始训练，共 {epochs} 个 epoch...\n")

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # 训练
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # 验证
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - t0

        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)
        history['epoch_time'].append(epoch_time)

        print(f"\nEpoch [{epoch}/{epochs}] "
              f"训练损失: {train_loss:.4f}  训练准确率: {train_acc:.2f}%  "
              f"验证损失: {val_loss:.4f}  验证准确率: {val_acc:.2f}%  "
              f"学习率: {current_lr:.6f}  耗时: {epoch_time:.1f}s")

        # 保存最优模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(save_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'dataset': dataset_name,
                'num_classes': config['num_classes'],
            }, best_model_path)
            print(f"  ✓ 最优模型已保存 (验证准确率: {best_val_acc:.2f}%)")

        # 每 10 个 epoch 保存检查点
        if epoch % 10 == 0:
            ckpt_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'history': history,
                'dataset': dataset_name,
            }, ckpt_path)
            print(f"  ✓ 检查点已保存: {ckpt_path}")

        # 每个 epoch 保存训练历史
        history_path = os.path.join(save_dir, 'training_history.json')
        save_training_history(history, history_path)

    # 训练结束，绘制学习曲线
    print(f"\n训练完成！最高验证准确率: {best_val_acc:.2f}%")
    plot_path = os.path.join(save_dir, 'learning_curves.png')
    plot_training_history(history, plot_path)
    print(f"学习曲线已保存至: {plot_path}")

    return history, best_val_acc


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='图像识别模型训练')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'mnist'], help='数据集名称')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=128, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001, help='初始学习率')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--data-dir', type=str, default='./data', help='数据目录')
    parser.add_argument('--save-dir', type=str, default='./checkpoints', help='保存目录')
    parser.add_argument('--num-workers', type=int, default=2, help='数据加载线程数')
    args = parser.parse_args()

    train(
        dataset_name=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        num_workers=args.num_workers,
    )

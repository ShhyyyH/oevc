"""
工具函数
- 训练指标记录
- 学习曲线绘制
- 结果可视化
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')   # 非交互式后端，适用于服务器环境
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 尝试设置中文字体
def _setup_chinese_font():
    """尝试配置中文字体，失败时使用英文标签"""
    candidates = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
    for font in candidates:
        try:
            prop = fm.FontProperties(family=font)
            plt.rcParams['font.family'] = prop.get_name()
            plt.rcParams['axes.unicode_minus'] = False
            return True
        except Exception:
            continue
    return False

_CN_FONT_AVAILABLE = _setup_chinese_font()


class AverageMeter:
    """计算并存储均值和当前值"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_accuracy(outputs, labels):
    """
    计算 Top-1 准确率（百分比）

    参数:
        outputs: 模型输出 logits，形状 (N, C)
        labels: 真实标签，形状 (N,)

    返回:
        准确率（0~100）
    """
    _, predicted = outputs.max(1)
    correct = predicted.eq(labels).sum().item()
    return 100.0 * correct / labels.size(0)


def save_training_history(history, path):
    """
    将训练历史保存为 JSON 文件

    参数:
        history: 包含 train_loss/train_acc/val_loss/val_acc 等列表的字典
        path: 保存路径
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def load_training_history(path):
    """
    从 JSON 文件加载训练历史

    参数:
        path: JSON 文件路径

    返回:
        history 字典
    """
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def plot_training_history(history, save_path=None, show=False):
    """
    绘制训练过程中的损失和准确率曲线

    参数:
        history: 训练历史字典，包含以下键：
                 - train_loss / val_loss
                 - train_acc  / val_acc
                 - lr（可选）
        save_path: 图像保存路径（None 则不保存）
        show: 是否调用 plt.show() 显示图像
    """
    epochs = range(1, len(history['train_loss']) + 1)
    has_lr = 'lr' in history and len(history['lr']) > 0

    n_plots = 3 if has_lr else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))

    # --- 损失曲线 ---
    ax = axes[0]
    ax.plot(epochs, history['train_loss'], 'b-o', markersize=3, label='Train Loss')
    ax.plot(epochs, history['val_loss'], 'r-o', markersize=3, label='Val Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- 准确率曲线 ---
    ax = axes[1]
    ax.plot(epochs, history['train_acc'], 'b-o', markersize=3, label='Train Acc')
    ax.plot(epochs, history['val_acc'], 'r-o', markersize=3, label='Val Acc')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- 学习率曲线（可选）---
    if has_lr:
        ax = axes[2]
        ax.plot(epochs, history['lr'], 'g-o', markersize=3, label='Learning Rate')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"学习曲线已保存至: {save_path}")

    if show:
        plt.show()

    plt.close(fig)


def visualize_predictions(images, labels, predictions, class_names, save_path=None, n=16):
    """
    可视化模型预测结果

    参数:
        images: numpy 数组，形状 (N, H, W, C) 或 (N, C, H, W)
        labels: 真实标签列表
        predictions: 预测标签列表
        class_names: 类别名称列表
        save_path: 保存路径（None 则不保存）
        n: 显示的样本数量
    """
    n = min(n, len(images))
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    if rows == 1 and cols == 1:
        axes_flat = [axes]
    else:
        axes_flat = axes.flatten()

    for i in range(n):
        img = images[i]
        # 转换 (C, H, W) -> (H, W, C)
        if img.ndim == 3 and img.shape[0] in (1, 3):
            img = np.transpose(img, (1, 2, 0))
        # 归一化到 [0, 1]
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        if img.shape[-1] == 1:
            img = img.squeeze(-1)

        ax = axes_flat[i]
        cmap = 'gray' if img.ndim == 2 else None
        ax.imshow(img, cmap=cmap)

        true_name = class_names[labels[i]] if labels[i] < len(class_names) else str(labels[i])
        pred_name = class_names[predictions[i]] if predictions[i] < len(class_names) else str(predictions[i])
        color = 'green' if labels[i] == predictions[i] else 'red'
        ax.set_title(f"True: {true_name}\nPred: {pred_name}", color=color, fontsize=8)
        ax.axis('off')

    # 隐藏多余的子图
    for i in range(n, len(axes_flat)):
        axes_flat[i].axis('off')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"预测结果已保存至: {save_path}")

    plt.close(fig)


def print_summary(history):
    """打印训练结果摘要"""
    if not history['val_acc']:
        print("暂无训练记录。")
        return

    best_epoch = int(np.argmax(history['val_acc'])) + 1
    best_acc = max(history['val_acc'])
    final_train_acc = history['train_acc'][-1]
    final_val_acc = history['val_acc'][-1]
    total_epochs = len(history['val_acc'])

    total_time = sum(history.get('epoch_time', [0]))

    print("\n========== 训练结果摘要 ==========")
    print(f"总训练轮数:    {total_epochs}")
    print(f"总训练时间:    {total_time:.1f} 秒")
    print(f"最终训练准确率: {final_train_acc:.2f}%")
    print(f"最终验证准确率: {final_val_acc:.2f}%")
    print(f"最优验证准确率: {best_acc:.2f}%  (第 {best_epoch} 轮)")
    print("==================================\n")

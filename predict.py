"""
图像识别预测脚本
加载训练好的模型对新图像进行预测
"""

import os
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

from model import get_model
from utils import visualize_predictions


# CIFAR-10 类别名称
CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

CIFAR10_CLASSES_CN = ['飞机', '汽车', '鸟', '猫', '鹿',
                      '狗', '青蛙', '马', '船', '卡车']

MNIST_CLASSES = [str(i) for i in range(10)]


def load_model(checkpoint_path, model_name='cifar', num_classes=10, device=None):
    """
    从检查点加载模型

    参数:
        checkpoint_path: 检查点文件路径
        model_name: 模型类型，'cifar' 或 'mnist'
        num_classes: 类别数量
        device: 计算设备

    返回:
        加载好权重的模型（eval 模式）
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = get_model(model_name, num_classes=num_classes)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 兼容不同的检查点格式
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        val_acc = checkpoint.get('val_acc')
        val_acc_str = f"{val_acc:.2f}%" if val_acc is not None else "N/A"
        print(f"模型已从检查点加载（第 {checkpoint.get('epoch', '?')} 轮，"
              f"验证准确率: {val_acc_str}）")
    else:
        model.load_state_dict(checkpoint)
        print("模型权重已加载")

    model = model.to(device)
    model.eval()
    return model


def predict_image(model, image_path, dataset='cifar10', device=None):
    """
    对单张图像进行预测

    参数:
        model: 已加载的模型
        image_path: 图像文件路径
        dataset: 数据集类型，决定预处理方式
        device: 计算设备

    返回:
        (预测类别索引, 置信度, 所有类别的概率)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 根据数据集选择预处理
    if dataset == 'cifar10':
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
    else:  # mnist
        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

    image = Image.open(image_path).convert('RGB' if dataset == 'cifar10' else 'L')
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = probabilities.max(1)

    return predicted_idx.item(), confidence.item(), probabilities.squeeze().cpu().numpy()


def predict_batch_from_dataset(model, dataset_name='cifar10', data_dir='./data',
                                n_samples=16, save_dir='./results', device=None):
    """
    从测试集随机抽取样本进行预测并可视化

    参数:
        model: 已加载的模型
        dataset_name: 数据集名称
        data_dir: 数据目录
        n_samples: 样本数量
        save_dir: 结果保存目录
        device: 计算设备
    """
    import torchvision

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(save_dir, exist_ok=True)

    if dataset_name == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        test_dataset = torchvision.datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=transform
        )
        class_names = CIFAR10_CLASSES_CN
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        test_dataset = torchvision.datasets.MNIST(
            root=data_dir, train=False, download=True, transform=transform
        )
        class_names = MNIST_CLASSES

    # 随机抽取样本
    indices = np.random.choice(len(test_dataset), n_samples, replace=False)
    images, true_labels, pred_labels = [], [], []

    model.eval()
    with torch.no_grad():
        for idx in indices:
            img, label = test_dataset[idx]
            img_tensor = img.unsqueeze(0).to(device)
            output = model(img_tensor)
            pred = output.argmax(1).item()

            images.append(img.numpy())
            true_labels.append(label)
            pred_labels.append(pred)

    # 可视化
    save_path = os.path.join(save_dir, 'predictions.png')
    visualize_predictions(
        np.array(images), true_labels, pred_labels,
        class_names, save_path=save_path, n=n_samples
    )

    # 计算准确率
    correct = sum(t == p for t, p in zip(true_labels, pred_labels))
    print(f"样本准确率: {correct}/{n_samples} = {100 * correct / n_samples:.1f}%")
    return true_labels, pred_labels


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='图像识别预测')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/best_model.pth',
                        help='模型检查点路径')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'mnist'], help='数据集')
    parser.add_argument('--data-dir', type=str, default='./data', help='数据目录')
    parser.add_argument('--image', type=str, default=None, help='单张图像路径（可选）')
    parser.add_argument('--n-samples', type=int, default=16, help='可视化样本数量')
    parser.add_argument('--save-dir', type=str, default='./results', help='结果保存目录')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 确定模型类型
    model_name = 'cifar' if args.dataset == 'cifar10' else 'mnist'
    model = load_model(args.checkpoint, model_name=model_name, num_classes=10, device=device)

    if args.image:
        # 单张图像预测
        pred_idx, confidence, probs = predict_image(model, args.image, args.dataset, device)
        class_names = CIFAR10_CLASSES_CN if args.dataset == 'cifar10' else MNIST_CLASSES
        print(f"\n预测结果: {class_names[pred_idx]}（置信度: {confidence * 100:.1f}%）")
        print("\n各类别概率:")
        for i, (name, prob) in enumerate(zip(class_names, probs)):
            bar = '█' * int(prob * 30)
            print(f"  {name:8s}: {prob * 100:5.1f}%  {bar}")
    else:
        # 批量预测
        predict_batch_from_dataset(
            model, args.dataset, args.data_dir, args.n_samples, args.save_dir, device
        )

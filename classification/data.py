"""所有分类模型共用的数据读取与增强协议。"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder


def classification_transform(image_size: int, training: bool, augmentation: bool):
    """建立分类变换。

    这里与 Ultralytics 分类加载器对齐：训练时随机裁取原图 90%--100% 面积，
    再做水平/垂直翻转和轻量亮度、对比度变化；验证和测试采用确定性缩放。
    所有张量只通过 ``ToTensor`` 缩放到 0--1，不再做 ImageNet mean/std
    标准化。这与当前 Ultralytics 分类数据管线实际送入 YOLO 的数值范围一致，
    避免树突模型和 YOLO 使用两套不同输入尺度。
    """
    operations: list = []
    if training and augmentation:
        operations.extend((
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.9, 1.0),
                ratio=(3.0 / 4.0, 4.0 / 3.0),
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
        ))
    else:
        # X-SDD 绝大多数原图为 128x128；这一写法也兼容少量非正方形图片，
        # 并与 Ultralytics 的 Resize(short edge) + CenterCrop 保持一致。
        operations.extend((transforms.Resize(image_size), transforms.CenterCrop(image_size)))
    operations.append(transforms.ToTensor())
    return transforms.Compose(operations)


def _seed_worker(worker_id: int) -> None:
    """为每个数据加载进程固定 NumPy/Python 随机种子。"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_classification_loaders(
    root: str | Path,
    image_size: int,
    batch_size: int,
    workers: int,
    device: torch.device,
    seed: int,
    augmentation: bool,
) -> tuple[dict[str, ImageFolder], dict[str, DataLoader]]:
    """读取标准 ``train/val/test/<类别>`` 目录并建立三个 DataLoader。"""
    root = Path(root).resolve()
    datasets = {
        "train": ImageFolder(root / "train", classification_transform(image_size, True, augmentation)),
        "val": ImageFolder(root / "val", classification_transform(image_size, False, False)),
        "test": ImageFolder(root / "test", classification_transform(image_size, False, False)),
    }
    classes = datasets["train"].classes
    if any(dataset.classes != classes for dataset in datasets.values()):
        raise ValueError("train、val、test 的类别目录不一致")

    loaders: dict[str, DataLoader] = {}
    for index, (split, dataset) in enumerate(datasets.items()):
        generator = torch.Generator().manual_seed(seed + index)
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=workers,
            pin_memory=device.type == "cuda",
            persistent_workers=workers > 0,
            worker_init_fn=_seed_worker,
            generator=generator,
        )
    return datasets, loaders

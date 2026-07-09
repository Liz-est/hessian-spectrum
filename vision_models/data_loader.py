import os 
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Subset
import matplotlib.pyplot as plt
from tqdm import tqdm


def data_loader(root, batch_size=256, workers=1, pin_memory=True,
                shuffle=True, mode="full", save_path=None):
    """
    mode:
        "full"     -> 使用全集 (默认)
        "uniform"  -> 每类固定采10张 (论文: Small ImageNet, ~10k总数)
        "1k"       -> 每类采样 ⌈1300/(k+1)⌉ 张 (论文: Heavy-Tailed ImageNet, ~10,217总数)
    save_path:
        如果不是 None，会保存子集分布直方图
    """

    traindir = os.path.join(root, 'train')
    valdir = os.path.join(root, 'val')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if shuffle:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    full_train_dataset = datasets.ImageFolder(traindir, train_transform)
    val_dataset = datasets.ImageFolder(valdir, val_transform)

    num_classes = len(full_train_dataset.classes)

    print("Data Sampling ...")
    if mode == "full":
        train_dataset = full_train_dataset

    else:
        # 每类索引
        labels = full_train_dataset.targets
        idx_per_class = [[] for _ in range(num_classes)]
        for idx, label in tqdm(enumerate(labels), total=len(labels), desc="Index sampling"):
            idx_per_class[label].append(idx)

        # 初始化 class_counts（在采样前）
        class_counts = {k: 0 for k in range(num_classes)}

        if mode == "uniform":
            # 每类固定采 10 张 (论文 Small ImageNet)
            indices = []
            for k in tqdm(range(num_classes), total=num_classes, desc="Uniform sampling"):
                candidates = idx_per_class[k]
                n_k = min(len(candidates), 10)
                if n_k > 0:
                    indices.extend(
                        np.random.choice(candidates, n_k, replace=False)
                    )
                class_counts[k] = n_k
            print(f"[Uniform] Total sampled images: {len(indices)}")

        elif mode == "1k":
            # Heavy-tailed: 第 k 类采 ceil(1300/(k+1)) 张 (论文 Heavy-Tailed ImageNet)
            indices = []
            for k in tqdm(range(num_classes), total=num_classes, desc="1/k sampling"):
                candidates = idx_per_class[k]
                n_k = min(len(candidates), int(np.ceil(1300.0 / (k+1))))
                if n_k > 0:
                    indices.extend(
                        np.random.choice(candidates, n_k, replace=False)
                    )
                class_counts[k] = n_k
            print(f"[1/k] Total sampled images: {len(indices)}")

        else:
            raise ValueError(f"Unsupported mode: {mode}")

        train_dataset = Subset(full_train_dataset, indices)

        # 画直方图并保存
        if save_path is not None:
            plt.figure(figsize=(12, 6))
            plt.bar(class_counts.keys(), class_counts.values())
            plt.xlabel("Class label")
            plt.ylabel("Number of samples")
            plt.title(f"Class distribution (mode={mode})")
            plt.tight_layout()
            plt.savefig(f'{save_path}/dataset_distribution.png', dpi=500)
            plt.close()
            print(f"[Subset mode={mode}] distribution plot saved to {save_path}")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=pin_memory,
        sampler=None,
        persistent_workers=True,
        prefetch_factor=2
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=True,
        prefetch_factor=2
    )
    return train_loader, val_loader

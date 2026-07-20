"""X-SDD 受控树突/普通卷积图像分类训练入口。"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from classification.data import build_classification_loaders
from classification.metrics import (
    benchmark_classifier,
    evaluate_classifier,
    metrics_from_predictions,
    public_metrics,
    save_confusion_matrix,
    save_predictions,
)
from classification.models import MODEL_VARIANTS, build_classifier


CONTROLLED_DATA = ROOT / "datasets" / "xsdd_yolo11_classification"
CSV_COLUMNS = (
    "epoch",
    "train_loss", "train_accuracy", "train_macro_precision", "train_macro_recall", "train_macro_f1",
    "val_loss", "val_accuracy", "val_macro_precision", "val_macro_recall", "val_macro_f1",
    "epoch_seconds", "elapsed_seconds", "gpu_memory_mb", "learning_rate",
)


def set_seed(seed: int) -> None:
    """固定所有随机源，使不同结构使用相同的可复现实验协议。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer,
    criterion: nn.Module,
    device: torch.device,
    class_names: list[str],
    description: str,
    show_progress: bool,
) -> dict:
    """训练一个 epoch，同时计算图像级分类指标。"""
    model.train()
    total_loss = 0.0
    targets: list[int] = []
    predictions: list[int] = []
    batches = tqdm(loader, desc=description, leave=False, dynamic_ncols=True, disable=not show_progress)
    for images, labels in batches:
        images = images.to(device, non_blocking=device.type == "cuda")
        labels = labels.to(device, non_blocking=device.type == "cuda")
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += float(loss.detach()) * images.shape[0]
        targets.extend(labels.detach().cpu().tolist())
        predictions.extend(logits.detach().argmax(dim=1).cpu().tolist())
        batches.set_postfix(loss=f"{float(loss.detach()):.4f}")
    metrics = metrics_from_predictions(targets, predictions, class_names)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


def csv_row(epoch: int, train: dict, val: dict, epoch_seconds: float, elapsed: float, memory: float, lr: float):
    return (
        epoch,
        train["loss"], train["accuracy"], train["macro_precision"], train["macro_recall"], train["macro_f1"],
        val["loss"], val["accuracy"], val["macro_precision"], val["macro_recall"], val["macro_f1"],
        epoch_seconds, elapsed, memory, lr,
    )


def main(
    default_variant: str = "v1",
    default_out: str | Path | None = None,
    default_branch_features: int = 4,
) -> None:
    """运行共享七分类协议；各独立入口只覆盖模型名称和输出目录。"""
    if default_variant not in MODEL_VARIANTS:
        raise ValueError(f"无效默认模型：{default_variant}")
    default_out = default_out or (ROOT / "runs1" / "controlled" / "xsdd_dnm_v1_cls")
    parser = argparse.ArgumentParser(description="X-SDD 树突/普通卷积受控分类训练")
    parser.add_argument("--variant", choices=MODEL_VARIANTS, default=default_variant)
    parser.add_argument("--data", default=str(CONTROLLED_DATA), help="包含 train/val/test 类别文件夹的根目录")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--branches", type=int, default=4)
    parser.add_argument("--branch-features", type=int, default=default_branch_features)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--out", default=str(default_out))
    parser.add_argument("--no-augmentation", action="store_true", help="关闭轻量几何和亮度增强")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--benchmark-iterations", type=int, default=200)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    datasets, loaders = build_classification_loaders(
        args.data,
        args.img_size,
        args.batch_size,
        args.workers,
        device,
        args.seed,
        augmentation=not args.no_augmentation,
    )
    class_names = datasets["train"].classes
    model = build_classifier(
        args.variant, len(class_names), args.width, args.branches, args.branch_features
    ).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0.0)

    output = Path(args.out).resolve()
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "metrics.csv"
    comparison_path = output / "comparison_metrics.csv"
    for path in (metrics_path, comparison_path):
        with path.open("w", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow(CSV_COLUMNS)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    protocol = {
        "protocol": "controlled_classification_v1",
        "task": "classification",
        "model": args.variant,
        "aggregation": {
            "v1": "legacy_direct_product",
            "v2a": "log_domain_exact_product",
            "v2b": "log_domain_geometric_mean",
            "conv": "parameter_matched_mlp",
        }[args.variant],
        "data": str(Path(args.data).resolve()),
        "class_names": class_names,
        "epochs": args.epochs,
        "img_size": args.img_size,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "pretrained": False,
        "augmentation": not args.no_augmentation,
        "optimizer": "AdamW",
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
        "scheduler": "CosineAnnealingLR(eta_min=0)",
        # YOLO 的 best.pt 依据验证 Top-1 Accuracy 选择；这里使用同一标准。
        "selection_metric": "validation_accuracy",
        "branches": args.branches,
        "branch_features": args.branch_features,
        "parameters": parameter_count,
        "trainable_parameters": trainable_count,
    }
    (output / "experiment_config.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print(
        f"任务：{Path(args.data).resolve().name} {len(class_names)} 分类；"
        f"模型：{args.variant}；输入：{args.img_size}×{args.img_size}"
    )
    print(f"设备：{device_name}；参数量：{parameter_count:,}；输出：{output}")
    print(f"类别：{', '.join(class_names)}")

    best_score = float("-inf")
    best_epoch = 0
    best_validation = None
    training_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        epoch_start = time.perf_counter()
        learning_rate = optimizer.param_groups[0]["lr"]
        train_metrics = train_one_epoch(
            model, loaders["train"], optimizer, criterion, device, class_names,
            f"训练 {epoch:03d}/{args.epochs}", not args.no_progress,
        )
        val_metrics = evaluate_classifier(
            model, loaders["val"], device, class_names, criterion
        )
        scheduler.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            gpu_memory_mb = torch.cuda.max_memory_allocated(device) / 1024**2
        else:
            gpu_memory_mb = 0.0
        epoch_seconds = time.perf_counter() - epoch_start
        elapsed_seconds = time.perf_counter() - training_start
        row = csv_row(
            epoch, train_metrics, val_metrics, epoch_seconds, elapsed_seconds,
            gpu_memory_mb, learning_rate,
        )
        for path in (metrics_path, comparison_path):
            with path.open("a", newline="", encoding="utf-8") as file:
                csv.writer(file).writerow(row)

        print(
            f"epoch {epoch:03d}/{args.epochs} "
            f"train(loss={train_metrics['loss']:.4f}, acc={train_metrics['accuracy']:.4f}, f1={train_metrics['macro_f1']:.4f}) "
            f"val(loss={val_metrics['loss']:.4f}, acc={val_metrics['accuracy']:.4f}, f1={val_metrics['macro_f1']:.4f}) "
            f"time={epoch_seconds:.1f}s gpu_mem={gpu_memory_mb:.0f}MB"
        )
        checkpoint = {
            "model": model.state_dict(),
            "variant": args.variant,
            "class_names": class_names,
            "width": args.width,
            "branches": args.branches,
            "branch_features": args.branch_features,
            "img_size": args.img_size,
            "epoch": epoch,
            "validation": public_metrics(val_metrics),
        }
        torch.save(checkpoint, output / "last.pt")
        if val_metrics["accuracy"] > best_score:
            best_score = float(val_metrics["accuracy"])
            best_epoch = epoch
            best_validation = public_metrics(val_metrics)
            torch.save(checkpoint, output / "best.pt")

    best_checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model"])
    test_metrics = evaluate_classifier(
        model, loaders["test"], device, class_names, criterion
    )
    save_confusion_matrix(output / "confusion_matrix.csv", test_metrics, class_names)
    save_predictions(output / "test_predictions.csv", datasets["test"], test_metrics)
    speed = benchmark_classifier(
        model, device, args.img_size, iterations=args.benchmark_iterations
    )
    checkpoint_size_mb = (output / "best.pt").stat().st_size / 1024**2
    summary = {
        **protocol,
        "best_epoch": best_epoch,
        "best_validation": best_validation,
        "test": public_metrics(test_metrics),
        "speed_batch1_forward": speed,
        "parameter_size_fp32_mb": parameter_count * 4 / 1024**2,
        "checkpoint_size_mb": checkpoint_size_mb,
    }
    (output / "test_metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("测试集最终结果：")
    print(json.dumps(summary["test"], indent=2, ensure_ascii=False))
    print(f"推理测速：{json.dumps(speed, ensure_ascii=False)}")


if __name__ == "__main__":
    main()

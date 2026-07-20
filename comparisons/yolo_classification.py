"""YOLO11n-cls 与 YOLO26n-cls 共用的表面缺陷分类训练实现。"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from classification.metrics import (
    benchmark_classifier,
    evaluate_classifier,
    public_metrics,
    save_confusion_matrix,
    save_predictions,
)
from comparisons.control import (
    controlled_classification_options,
    standardize_yolo_classification_metrics,
    write_protocol,
)


ROOT = Path(__file__).resolve().parents[1]
CONTROLLED_DATA = ROOT / "datasets" / "xsdd_yolo11_classification"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _best_validation(results_csv: Path) -> tuple[int | None, dict | None]:
    """从 Ultralytics CSV 中读取验证 Top-1 最佳轮次。"""
    with results_csv.open("r", newline="", encoding="utf-8-sig") as file:
        rows = [{key.strip(): value.strip() for key, value in row.items() if key} for row in csv.DictReader(file)]
    candidates = []
    for row in rows:
        value = row.get("metrics/accuracy_top1", row.get("metrics/accuracy_top1(B)", ""))
        if value:
            candidates.append((float(value), row))
    if not candidates:
        return None, None
    accuracy, row = max(candidates, key=lambda item: item[0])
    epoch = int(float(row["epoch"]))
    if rows and int(float(rows[0]["epoch"])) == 0:
        epoch += 1
    return epoch, {
        "accuracy": accuracy,
        "loss": float(row["val/loss"]) if row.get("val/loss", "") else None,
    }


def _build_yolo_eval_loaders(
    root: str | Path,
    image_size: int,
    batch_size: int,
    workers: int,
    device: torch.device,
) -> tuple[dict[str, ImageFolder], dict[str, DataLoader]]:
    """使用当前 Ultralytics 版本真实的分类验证预处理。

    本机 Ultralytics 8.4.96 的 ``classify_transforms`` 为 Resize、CenterCrop、
    ToTensor，即像素保持在 0--1；这里直接复用它，避免再次手写错误归一化。
    """
    from ultralytics.data.augment import classify_transforms

    root = Path(root).resolve()
    transform = classify_transforms(image_size)
    datasets = {
        split: ImageFolder(root / split, transform=transform)
        for split in ("train", "val", "test")
    }
    class_names = datasets["train"].classes
    if any(dataset.classes != class_names for dataset in datasets.values()):
        raise ValueError("train、val、test 的类别目录不一致")
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=device.type == "cuda",
            persistent_workers=workers > 0,
        )
        for split, dataset in datasets.items()
    }
    return datasets, loaders


def main(
    version: str,
    default_pretrained: bool = True,
    default_data: str | Path | None = None,
    default_run_prefix: str = "",
) -> None:
    """训练指定版本的 YOLO nano 分类模型。"""
    if version not in {"11", "26"}:
        raise ValueError("version 只能是 11 或 26")
    dataset_default = Path(default_data) if default_data is not None else CONTROLLED_DATA
    parser = argparse.ArgumentParser(description=f"YOLO{version}n-cls：图像分类基线")
    parser.add_argument("--data", default=str(dataset_default))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument(
        "--scratch",
        action="store_true",
        help="从 YAML 随机初始化",
    )
    initialization.add_argument(
        "--pretrained",
        action="store_true",
        help="显式加载 ImageNet 分类预训练权重",
    )
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument("--fraction", type=float, default=1.0, help="仅供冒烟测试；正式实验必须为1.0")
    parser.add_argument("--benchmark-iterations", type=int, default=200)
    parser.add_argument(
        "--run-name",
        default=None,
        help="自定义 runs1/controlled 下的结果目录名",
    )
    args = parser.parse_args()
    if not 0 < args.fraction <= 1:
        raise ValueError("--fraction 必须位于 (0, 1]")

    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise SystemExit("缺少 ultralytics，请在当前 pytorch 环境中安装。") from error

    set_seed(args.seed)
    pretrained = default_pretrained
    if args.scratch:
        pretrained = False
    elif args.pretrained:
        pretrained = True
    model_source = f"yolo{version}n-cls.pt" if pretrained else f"yolo{version}n-cls.yaml"
    base_run_name = f"yolo{version}n_cls" if pretrained else f"yolo{version}n_cls_scratch"
    if default_run_prefix:
        base_run_name = f"{default_run_prefix}_{base_run_name}"
    run_name = args.run_name or base_run_name
    data_path = Path(args.data).resolve()
    train_root = data_path / "train"
    class_names_on_disk = sorted(path.name for path in train_root.iterdir() if path.is_dir())
    if not class_names_on_disk:
        raise ValueError(f"训练目录中没有类别子文件夹：{train_root}")
    print(
        f"任务：{data_path.name} {len(class_names_on_disk)} 分类；"
        f"模型：{model_source}；输入：{args.img_size}×{args.img_size}"
    )
    print(f"初始化：{'ImageNet 预训练' if pretrained else '随机初始化'}；结果目录：runs1/controlled/{run_name}")

    model = YOLO(model_source)
    gpu_memory_mb_by_epoch: dict[int, float] = {}
    validation_metrics_by_epoch: dict[int, dict[str, float]] = {}
    epoch_seconds_by_epoch: dict[int, float] = {}
    callback_state: dict[str, object] = {"epoch_start": None, "val_loader": None, "class_names": None}

    def reset_peak_memory(trainer) -> None:
        callback_state["epoch_start"] = time.perf_counter()
        if trainer.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(trainer.device)

    def collect_unified_validation(trainer) -> None:
        """每轮用项目共用评估器补齐 YOLO 原生日志没有的宏平均指标。"""
        epoch = trainer.epoch + 1
        # Ultralytics 在训练全部完成后的最终复评还会再次触发同一 callback，
        # 那不是新的训练 epoch，不能重复写入或打印成 epochs+1。
        if epoch > args.epochs or epoch in validation_metrics_by_epoch:
            return
        if callback_state["val_loader"] is None:
            callback_datasets, callback_loaders = _build_yolo_eval_loaders(
                args.data,
                args.img_size,
                args.batch_size,
                args.workers,
                trainer.device,
            )
            callback_state["val_loader"] = callback_loaders["val"]
            callback_state["class_names"] = callback_datasets["train"].classes
        # Ultralytics 内置验证和 best.pt 都基于 EMA 权重；统一指标也必须评估
        # 同一组权重，否则同一轮的 Top-1 和 Macro-F1 会来自两个不同模型。
        evaluation_model = trainer.ema.ema if trainer.ema is not None else trainer.model
        unified = evaluate_classifier(
            evaluation_model,
            callback_state["val_loader"],
            trainer.device,
            callback_state["class_names"],
            nn.CrossEntropyLoss(),
        )
        validation_metrics_by_epoch[epoch] = public_metrics(unified)
        if trainer.device.type == "cuda":
            gpu_memory_mb_by_epoch[epoch] = (
                torch.cuda.max_memory_allocated(trainer.device) / 1024**2
            )
        epoch_start = callback_state.get("epoch_start")
        if isinstance(epoch_start, float):
            epoch_seconds_by_epoch[epoch] = time.perf_counter() - epoch_start
        print(
            f"统一验证 epoch {epoch:03d}: "
            f"acc={unified['accuracy']:.4f}, "
            f"macro_P={unified['macro_precision']:.4f}, "
            f"macro_R={unified['macro_recall']:.4f}, "
            f"macro_F1={unified['macro_f1']:.4f}"
        )

    model.add_callback("on_train_epoch_start", reset_peak_memory)
    model.add_callback("on_fit_epoch_end", collect_unified_validation)
    controlled = controlled_classification_options(args.epochs, not args.no_augmentation)
    controlled["workers"] = args.workers
    training_start = time.perf_counter()
    model.train(
        data=str(Path(args.data).resolve()),
        epochs=args.epochs,
        imgsz=args.img_size,
        batch=args.batch_size,
        device=args.device,
        seed=args.seed,
        pretrained=pretrained,
        fraction=args.fraction,
        project=str(ROOT / "runs1" / "controlled"),
        name=run_name,
        exist_ok=True,
        **controlled,
    )
    training_seconds = time.perf_counter() - training_start
    save_dir = Path(model.trainer.save_dir)
    standardize_yolo_classification_metrics(
        save_dir / "results.csv",
        save_dir / "comparison_metrics.csv",
        gpu_memory_mb_by_epoch,
        validation_metrics_by_epoch,
        epoch_seconds_by_epoch,
    )

    best_wrapper = YOLO(save_dir / "weights" / "best.pt")
    raw_model = best_wrapper.model
    device = model.trainer.device
    datasets, loaders = _build_yolo_eval_loaders(
        args.data,
        args.img_size,
        args.batch_size,
        args.workers,
        device,
    )
    class_names = datasets["train"].classes
    if isinstance(raw_model.names, dict):
        trained_names = [raw_model.names[index] for index in sorted(raw_model.names)]
    else:
        trained_names = list(raw_model.names)
    if trained_names != class_names:
        raise ValueError(f"YOLO 类别顺序 {trained_names} 与数据加载器 {class_names} 不一致")

    criterion = nn.CrossEntropyLoss()
    test_metrics = evaluate_classifier(
        raw_model.to(device), loaders["test"], device, class_names, criterion
    )
    save_confusion_matrix(save_dir / "confusion_matrix.csv", test_metrics, class_names)
    save_predictions(save_dir / "test_predictions.csv", datasets["test"], test_metrics)
    speed = benchmark_classifier(
        raw_model,
        device,
        args.img_size,
        iterations=args.benchmark_iterations,
    )
    parameter_count = sum(parameter.numel() for parameter in raw_model.parameters())
    # Ultralytics 保存/重载 best.pt 时会把 requires_grad 全部置 False；本实验没有
    # freeze 参数，训练阶段实际可训练参数就是全部参数。
    trainable_count = parameter_count
    best_epoch, best_validation = _best_validation(save_dir / "results.csv")
    protocol = {
        "protocol": "controlled_classification_v1",
        "task": "classification",
        "model": f"yolo{version}n-cls",
        "model_source": model_source,
        "data": str(Path(args.data).resolve()),
        "class_names": class_names,
        "epochs": args.epochs,
        "img_size": args.img_size,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "pretrained": pretrained,
        "augmentation": not args.no_augmentation,
        "fraction": args.fraction,
        "parameters": parameter_count,
        "trainable_parameters": trainable_count,
        "training_seconds": training_seconds,
        **controlled,
    }
    write_protocol(save_dir, protocol)
    checkpoint_size_mb = (save_dir / "weights" / "best.pt").stat().st_size / 1024**2
    summary = {
        **protocol,
        "best_epoch": best_epoch,
        "best_validation": best_validation,
        "test": public_metrics(test_metrics),
        "speed_batch1_forward": speed,
        "parameter_size_fp32_mb": parameter_count * 4 / 1024**2,
        "checkpoint_size_mb": checkpoint_size_mb,
    }
    (save_dir / "test_metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("测试集最终结果：")
    print(json.dumps(summary["test"], indent=2, ensure_ascii=False))
    print(f"推理测速：{json.dumps(speed, ensure_ascii=False)}")
    print(f"统一逐轮指标：{save_dir / 'comparison_metrics.csv'}")

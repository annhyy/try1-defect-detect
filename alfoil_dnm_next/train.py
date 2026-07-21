"""Training runner for isolated X-SDD F4 and V1-tuned experiments."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alfoil_dnm_next.models import (
    BRANCH_FEATURES,
    MODEL_VARIANTS as DNM_MODEL_VARIANTS,
    build_classifier,
)
from classification.data import build_classification_loaders
from classification.metrics import (
    benchmark_classifier,
    evaluate_classifier,
    metrics_from_predictions,
    public_metrics,
    save_confusion_matrix,
    save_predictions,
)
from classification.models import build_classifier as build_control_classifier


CONTROLLED_DATA = ROOT / "datasets" / "xsdd_yolo11_classification"
TRAIN_VARIANTS = (*DNM_MODEL_VARIANTS, "conv_control")


@dataclass(frozen=True)
class ExperimentSpec:
    display_name: str
    aggregation: str
    epochs: int
    backbone_lr: float
    head_lr: float
    min_lr_ratio: float
    output_name: str


EXPERIMENTS = {
    "v2a_f4": ExperimentSpec(
        display_name="DNM-V2a-F4",
        aggregation="log_domain_exact_product_f4",
        epochs=100,
        backbone_lr=1e-3,
        head_lr=1e-3,
        min_lr_ratio=0.0,
        output_name="xsdd_dnm_v2a_f4_cls",
    ),
    "v2b_f4": ExperimentSpec(
        display_name="DNM-V2b-F4",
        aggregation="log_domain_geometric_mean_f4",
        epochs=100,
        backbone_lr=1e-3,
        head_lr=1e-3,
        min_lr_ratio=0.0,
        output_name="xsdd_dnm_v2b_f4_cls",
    ),
    "v1_tuned": ExperimentSpec(
        display_name="DNM-V1-Tuned",
        aggregation="branch_specific_log_product_f4_signed_output",
        epochs=150,
        backbone_lr=1e-3,
        head_lr=3e-3,
        min_lr_ratio=0.01,
        output_name="xsdd_dnm_v1_tuned_cls",
    ),
    "conv_control": ExperimentSpec(
        display_name="Conv-Control-Weighted",
        aggregation="parameter_matched_mlp",
        epochs=150,
        backbone_lr=1e-3,
        head_lr=1e-3,
        min_lr_ratio=0.01,
        output_name="xsdd_conv_control_unweighted_150e_cls",
    ),
}


CSV_COLUMNS = (
    "epoch",
    "train_loss",
    "train_accuracy",
    "train_macro_precision",
    "train_macro_recall",
    "train_macro_f1",
    "train_pred_count",
    "val_loss",
    "val_accuracy",
    "val_macro_precision",
    "val_macro_recall",
    "val_macro_f1",
    "val_pred_count",
    "epoch_seconds",
    "elapsed_seconds",
    "gpu_memory_mb",
    "learning_rate",
    "head_learning_rate",
)


def set_seed(seed: int) -> None:
    """Fix random sources to retain the controlled comparison protocol."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def add_prediction_counts(metrics: dict) -> dict:
    """Attach class-wise prediction totals derived from the confusion matrix."""
    matrix = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    metrics["pred_count"] = matrix.sum(axis=0).tolist()
    return metrics


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
    model.train()
    total_loss = 0.0
    targets: list[int] = []
    predictions: list[int] = []
    batches = tqdm(
        loader,
        desc=description,
        leave=False,
        dynamic_ncols=True,
        disable=not show_progress,
    )
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
    metrics["loss"] = total_loss / max(len(loader.dataset), 1)
    return add_prediction_counts(metrics)


def balanced_class_weights(targets: list[int], classes: int) -> Tensor:
    """Use N/(C*n_c), the standard inverse-frequency balanced weights."""
    counts = torch.bincount(torch.as_tensor(targets), minlength=classes).float()
    if torch.any(counts == 0):
        raise ValueError("Balanced loss requires at least one training sample per class")
    return counts.sum() / (classes * counts)


def build_optimizer_and_scheduler(
    model: nn.Module,
    epochs: int,
    backbone_lr: float,
    head_lr: float,
    min_lr_ratio: float,
    weight_decay: float,
):
    """Preserve the old single-LR F4 optimizer and use two LRs for V1-Tuned."""
    if math.isclose(backbone_lr, head_lr, rel_tol=0.0, abs_tol=0.0):
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=backbone_lr, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, epochs, eta_min=backbone_lr * min_lr_ratio
        )
    else:
        optimizer = torch.optim.AdamW(
            (
                {"params": model.backbone.parameters(), "lr": backbone_lr},
                {"params": model.head.parameters(), "lr": head_lr},
            ),
            weight_decay=weight_decay,
        )

        def cosine_factor(step: int) -> float:
            cosine = 0.5 * (1.0 + math.cos(math.pi * step / epochs))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=(cosine_factor, cosine_factor)
        )
    return optimizer, scheduler


def current_learning_rates(optimizer) -> tuple[float, float]:
    backbone_lr = float(optimizer.param_groups[0]["lr"])
    head_lr = (
        float(optimizer.param_groups[1]["lr"])
        if len(optimizer.param_groups) > 1
        else backbone_lr
    )
    return backbone_lr, head_lr


def csv_row(
    epoch: int,
    train: dict,
    val: dict,
    epoch_seconds: float,
    elapsed: float,
    memory: float,
    backbone_lr: float,
    head_lr: float,
) -> tuple:
    return (
        epoch,
        train["loss"],
        train["accuracy"],
        train["macro_precision"],
        train["macro_recall"],
        train["macro_f1"],
        json.dumps(train["pred_count"], separators=(",", ":")),
        val["loss"],
        val["accuracy"],
        val["macro_precision"],
        val["macro_recall"],
        val["macro_f1"],
        json.dumps(val["pred_count"], separators=(",", ":")),
        epoch_seconds,
        elapsed,
        memory,
        backbone_lr,
        head_lr,
    )


def parse_args(default_variant: str, default_class_weighting: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="X-SDD DNM F4 ablations and V1-Tuned classification"
    )
    parser.add_argument("--variant", choices=TRAIN_VARIANTS, default=default_variant)
    parser.add_argument("--data", default=str(CONTROLLED_DATA))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--branches", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backbone-learning-rate", type=float)
    parser.add_argument("--head-learning-rate", type=float)
    parser.add_argument("--min-learning-rate-ratio", type=float)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument(
        "--class-weighting",
        choices=("none", "balanced"),
        default=default_class_weighting,
        help="Run the separate inverse-frequency weighted-loss experiment",
    )
    parser.add_argument("--out")
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--benchmark-iterations", type=int, default=200)
    parser.add_argument("--skip-benchmark", action="store_true")
    return parser.parse_args()


def resolved_output(args: argparse.Namespace, spec: ExperimentSpec) -> Path:
    if args.out:
        return Path(args.out).resolve()
    if args.variant == "conv_control" and args.class_weighting == "balanced":
        output_name = "xsdd_conv_control_weighted_cls"
    else:
        output_name = spec.output_name
    if args.variant != "conv_control" and args.class_weighting == "balanced":
        output_name = output_name.removesuffix("_cls") + "_weighted_cls"
    return (ROOT / "runs1" / "controlled" / output_name).resolve()


def main(
    default_variant: str = "v1_tuned",
    default_class_weighting: str = "none",
) -> None:
    if default_variant not in TRAIN_VARIANTS:
        raise ValueError(f"Invalid default model: {default_variant}")
    if default_class_weighting not in {"none", "balanced"}:
        raise ValueError(f"Invalid default class weighting: {default_class_weighting}")
    args = parse_args(default_variant, default_class_weighting)
    spec = EXPERIMENTS[args.variant]
    epochs = args.epochs if args.epochs is not None else spec.epochs
    backbone_lr = (
        args.backbone_learning_rate
        if args.backbone_learning_rate is not None
        else spec.backbone_lr
    )
    head_lr = (
        args.head_learning_rate
        if args.head_learning_rate is not None
        else spec.head_lr
    )
    min_lr_ratio = (
        args.min_learning_rate_ratio
        if args.min_learning_rate_ratio is not None
        else spec.min_lr_ratio
    )
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if backbone_lr <= 0 or head_lr <= 0:
        raise ValueError("learning rates must be positive")
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError("min-learning-rate-ratio must be between 0 and 1")
    if args.benchmark_iterations < 1 and not args.skip_benchmark:
        raise ValueError("benchmark-iterations must be at least 1")

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
    branch_features = 8 if args.variant == "conv_control" else BRANCH_FEATURES
    if args.variant == "conv_control":
        model = build_control_classifier(
            "conv", len(class_names), args.width, args.branches, branch_features
        ).to(device)
    else:
        model = build_classifier(
            args.variant, len(class_names), args.width, args.branches
        ).to(device)

    class_weights = None
    if args.class_weighting == "balanced":
        class_weights = balanced_class_weights(
            datasets["train"].targets, len(class_names)
        ).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights, label_smoothing=args.label_smoothing
    )
    optimizer, scheduler = build_optimizer_and_scheduler(
        model,
        epochs,
        backbone_lr,
        head_lr,
        min_lr_ratio,
        args.weight_decay,
    )

    output = resolved_output(args, spec)
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "metrics.csv"
    comparison_path = output / "comparison_metrics.csv"
    for path in (metrics_path, comparison_path):
        with path.open("w", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow(CSV_COLUMNS)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    backbone_parameters = sum(
        parameter.numel() for parameter in model.backbone.parameters()
    )
    head_parameters = sum(parameter.numel() for parameter in model.head.parameters())
    scheduler_name = (
        f"CosineAnnealingLR(eta_min_ratio={min_lr_ratio:g})"
        if len(optimizer.param_groups) == 1
        else f"CosineLambdaLR(min_ratio={min_lr_ratio:g})"
    )
    protocol = {
        "protocol": "controlled_classification_dnm_next_v1",
        "task": "classification",
        "model": args.variant,
        "display_name": spec.display_name,
        "aggregation": spec.aggregation,
        "data": str(Path(args.data).resolve()),
        "class_names": class_names,
        "epochs": epochs,
        "img_size": args.img_size,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "pretrained": False,
        "augmentation": not args.no_augmentation,
        "optimizer": "AdamW",
        "backbone_learning_rate": backbone_lr,
        "head_learning_rate": head_lr,
        "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
        "class_weighting": args.class_weighting,
        "class_weights": class_weights.detach().cpu().tolist() if class_weights is not None else None,
        "scheduler": scheduler_name,
        "minimum_learning_rate_ratio": min_lr_ratio,
        "selection_metrics": ["validation_accuracy", "validation_macro_f1"],
        "primary_checkpoint": "best_accuracy.pt",
        "branches": args.branches,
        "branch_features": branch_features,
        "branch_specific_projection": args.variant == "v1_tuned",
        "parameters": parameter_count,
        "trainable_parameters": trainable_count,
        "backbone_parameters": backbone_parameters,
        "head_parameters": head_parameters,
    }
    (output / "experiment_config.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print(
        f"Task: {Path(args.data).resolve().name} {len(class_names)} classes; "
        f"model: {spec.display_name}; input: {args.img_size}x{args.img_size}"
    )
    print(
        f"Device: {device_name}; parameters: {parameter_count:,} "
        f"(backbone={backbone_parameters:,}, head={head_parameters:,}); output: {output}"
    )
    print(f"Classes: {', '.join(class_names)}")
    if class_weights is not None:
        print(f"Class weights: {class_weights.detach().cpu().tolist()}")

    best_accuracy = float("-inf")
    best_accuracy_epoch = 0
    best_accuracy_validation = None
    best_macro_f1 = float("-inf")
    best_macro_f1_epoch = 0
    best_macro_f1_validation = None
    training_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        epoch_start = time.perf_counter()
        epoch_backbone_lr, epoch_head_lr = current_learning_rates(optimizer)
        train_metrics = train_one_epoch(
            model,
            loaders["train"],
            optimizer,
            criterion,
            device,
            class_names,
            f"Train {epoch:03d}/{epochs}",
            not args.no_progress,
        )
        val_metrics = add_prediction_counts(
            evaluate_classifier(model, loaders["val"], device, class_names, criterion)
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
            epoch,
            train_metrics,
            val_metrics,
            epoch_seconds,
            elapsed_seconds,
            gpu_memory_mb,
            epoch_backbone_lr,
            epoch_head_lr,
        )
        for path in (metrics_path, comparison_path):
            with path.open("a", newline="", encoding="utf-8") as file:
                csv.writer(file).writerow(row)

        print(
            f"epoch {epoch:03d}/{epochs} "
            f"train(loss={train_metrics['loss']:.4f}, acc={train_metrics['accuracy']:.4f}, "
            f"f1={train_metrics['macro_f1']:.4f}) "
            f"val(loss={val_metrics['loss']:.4f}, acc={val_metrics['accuracy']:.4f}, "
            f"f1={val_metrics['macro_f1']:.4f}) "
            f"pred_count={val_metrics['pred_count']} "
            f"lr=({epoch_backbone_lr:.3g},{epoch_head_lr:.3g}) "
            f"time={epoch_seconds:.1f}s gpu_mem={gpu_memory_mb:.0f}MB"
        )
        checkpoint = {
            "model": model.state_dict(),
            "variant": args.variant,
            "class_names": class_names,
            "width": args.width,
            "branches": args.branches,
            "branch_features": branch_features,
            "img_size": args.img_size,
            "epoch": epoch,
            "validation": public_metrics(val_metrics),
        }
        torch.save(checkpoint, output / "last.pt")
        if val_metrics["accuracy"] > best_accuracy:
            best_accuracy = float(val_metrics["accuracy"])
            best_accuracy_epoch = epoch
            best_accuracy_validation = public_metrics(val_metrics)
            torch.save(checkpoint, output / "best_accuracy.pt")
            torch.save(checkpoint, output / "best.pt")
        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = float(val_metrics["macro_f1"])
            best_macro_f1_epoch = epoch
            best_macro_f1_validation = public_metrics(val_metrics)
            torch.save(checkpoint, output / "best_macro_f1.pt")

    best_checkpoint = torch.load(
        output / "best_accuracy.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(best_checkpoint["model"])
    test_metrics = add_prediction_counts(
        evaluate_classifier(model, loaders["test"], device, class_names, criterion)
    )
    save_confusion_matrix(output / "confusion_matrix.csv", test_metrics, class_names)
    save_predictions(output / "test_predictions.csv", datasets["test"], test_metrics)
    speed = (
        {}
        if args.skip_benchmark
        else benchmark_classifier(
            model, device, args.img_size, iterations=args.benchmark_iterations
        )
    )
    checkpoint_size_mb = (output / "best_accuracy.pt").stat().st_size / 1024**2
    summary = {
        **protocol,
        "best_epoch": best_accuracy_epoch,
        "best_validation": best_accuracy_validation,
        "best_accuracy_checkpoint": {
            "path": "best_accuracy.pt",
            "epoch": best_accuracy_epoch,
            "validation": best_accuracy_validation,
        },
        "best_macro_f1_checkpoint": {
            "path": "best_macro_f1.pt",
            "epoch": best_macro_f1_epoch,
            "validation": best_macro_f1_validation,
        },
        "test_checkpoint": "best_accuracy.pt",
        "test": public_metrics(test_metrics),
        "speed_batch1_forward": speed,
        "parameter_size_fp32_mb": parameter_count * 4 / 1024**2,
        "checkpoint_size_mb": checkpoint_size_mb,
    }
    (output / "test_metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("Final test result (best validation Accuracy checkpoint):")
    print(json.dumps(summary["test"], indent=2, ensure_ascii=False))
    print(f"Inference benchmark: {json.dumps(speed, ensure_ascii=False)}")


if __name__ == "__main__":
    main()

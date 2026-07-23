"""训练以空间树突运算替换卷积的 APSPC 目标检测器。"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alfoil_dendritic_conv.model import (
    DendriticConvDetector,
    PlainConvDetector,
    SpatialDendriticConv2d,
)
from alfoil_dnm.data import YoloDefectDataset, load_data_yaml
from alfoil_dnm.metrics import evaluate_detection
from detection.dnm_train import (
    COMPARISON_COLUMNS,
    CONTROLLED_DATA,
    METRIC_COLUMNS,
    make_loader,
    run_epoch,
    set_seed,
)


DENDRITIC_COLUMNS = (
    "epoch", "layer", "weight_abs_mean", "threshold_mean", "distance_mean",
    "soma_slope_mean", "soma_threshold_mean", "synapse_grad_norm",
)


def append_row(path: Path, row: tuple) -> None:
    """向逐轮指标文件追加一行。"""
    with path.open("a", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow(row)


@torch.no_grad()
def dendritic_statistics(model: torch.nn.Module, epoch: int) -> list[tuple]:
    """汇总各树突替换层的参数尺度和最后一批突触梯度。"""
    rows = []
    layer_index = 0
    for module in model.modules():
        if not isinstance(module, SpatialDendriticConv2d):
            continue
        weight, threshold, distance = module.synapse.biological_parameters()
        gradients = [
            parameter.grad.reshape(-1)
            for parameter in (
                module.synapse.raw_weight,
                module.synapse.raw_threshold,
                module.synapse.raw_distance,
            )
            if parameter.grad is not None
        ]
        gradient_norm = (
            float(torch.cat(gradients).norm()) if gradients else 0.0
        )
        slope = F.softplus(module.soma.raw_slope) + 1e-4
        rows.append((
            epoch,
            layer_index,
            float(weight.abs().mean()),
            float(threshold.mean()),
            float(distance.mean()),
            float(slope.mean()),
            float(module.soma.threshold.mean()),
            gradient_norm,
        ))
        layer_index += 1
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="APSPC 空间树突卷积目标检测训练")
    parser.add_argument("--data", default=str(CONTROLLED_DATA), help="包含 train/val/test 的 YOLO data.yaml")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--variant", choices=("dendritic", "conv"), default="dendritic")
    parser.add_argument("--branches", type=int, default=4)
    parser.add_argument("--replace-layers", type=int, choices=(1, 2), default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument("--out", default=None)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    if args.eval_interval < 1:
        raise ValueError("--eval-interval 必须大于等于 1")

    set_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    config = load_data_yaml(args.data)
    cached_size = config.get("letterbox_size")
    if cached_size is not None and int(cached_size) != args.img_size:
        raise ValueError(
            f"数据已预先 letterbox 为 {cached_size}，不能再按 {args.img_size} 二次缩放"
        )

    train_set = YoloDefectDataset(config, "train", args.img_size)
    val_set = YoloDefectDataset(config, "val", args.img_size)
    test_set = YoloDefectDataset(config, "test", args.img_size)
    train_loader = make_loader(train_set, args.batch_size, args.workers, True, device, args.seed)
    val_loader = make_loader(val_set, args.batch_size, args.workers, False, device, args.seed)
    test_loader = make_loader(test_set, args.batch_size, args.workers, False, device, args.seed)

    if args.variant == "dendritic":
        model = DendriticConvDetector(
            len(config["names"]), args.width, args.branches, args.replace_layers
        ).to(device)
        model_name = f"dendritic_conv_r{args.replace_layers}"
    else:
        model = PlainConvDetector(len(config["names"]), args.width).to(device)
        model_name = "dendritic_conv_control"
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs, eta_min=0.0
    )
    output = Path(
        args.out or ROOT / "run2" / "controlled" / model_name
    ).resolve()
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "metrics.csv"
    comparison_path = output / "comparison_metrics.csv"
    dendritic_path = output / "dendritic_stats.csv" if args.variant == "dendritic" else None
    metrics_path.write_text(",".join(METRIC_COLUMNS) + "\n", encoding="utf-8")
    comparison_path.write_text(",".join(COMPARISON_COLUMNS) + "\n", encoding="utf-8")
    if dendritic_path is not None:
        dendritic_path.write_text(",".join(DENDRITIC_COLUMNS) + "\n", encoding="utf-8")

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    protocol = {
        "protocol": "apspc_detection_spatial_dendritic_conv_v1",
        "task": "object_detection",
        "model": model_name,
        "data": str(Path(args.data).resolve()),
        "class_names": config["names"],
        "epochs": args.epochs,
        "img_size": args.img_size,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "pretrained": False,
        "augmentation": "none",
        "optimizer": "AdamW",
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "scheduler": "CosineAnnealingLR(eta_min=0)",
        "amp": False,
        "parameters": parameter_count,
        "width": args.width,
        "branches": args.branches if args.variant == "dendritic" else None,
        "replace_layers": args.replace_layers if args.variant == "dendritic" else 0,
        "dendritic_kernel": "2x2" if args.variant == "dendritic" else None,
        "dendritic_stride": 1 if args.variant == "dendritic" else None,
        "branch_product_items": 4 if args.variant == "dendritic" else None,
        "padding": "replicate_right_bottom" if args.variant == "dendritic" else None,
        "dendritic_statistics": DENDRITIC_COLUMNS if args.variant == "dendritic" else None,
        "metric_selection": "validation_map50_95",
    }
    (output / "experiment_config.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"任务：APSPC 目标检测；模型：{model_name}；输入："
        f"{args.img_size}x{args.img_size}"
    )
    branch_text = f"；分支数：{args.branches}" if args.variant == "dendritic" else ""
    print(f"设备：{device}；参数量：{parameter_count:,}{branch_text}；输出：{output}")

    best_score = float("-inf")
    best_epoch = 0
    best_validation = None
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        epoch_started = time.perf_counter()
        learning_rate = optimizer.param_groups[0]["lr"]
        train_metrics = run_epoch(
            model, train_loader, optimizer, len(config["names"]), device,
            not args.no_progress, f"train {epoch:03d}/{args.epochs}",
        )
        val_metrics = run_epoch(
            model, val_loader, None, len(config["names"]), device,
            not args.no_progress, f"val {epoch:03d}/{args.epochs}",
        )
        should_evaluate = epoch % args.eval_interval == 0 or epoch == args.epochs
        detection = (
            evaluate_detection(model, val_loader, len(config["names"]), device, config["names"])
            if should_evaluate else None
        )
        scheduler.step()
        gpu_memory = (
            torch.cuda.max_memory_allocated(device) / 1024 ** 2
            if device.type == "cuda" else 0.0
        )
        epoch_seconds = time.perf_counter() - epoch_started
        elapsed_seconds = time.perf_counter() - started
        losses = (
            epoch,
            train_metrics["loss"], train_metrics["obj"], train_metrics["box"], train_metrics["cls"],
            val_metrics["loss"], val_metrics["obj"], val_metrics["box"], val_metrics["cls"],
        )
        detection_values = (
            ("", "", "", "") if detection is None
            else tuple(detection[key] for key in ("precision", "recall", "map50", "map50_95"))
        )
        runtime = (epoch_seconds, elapsed_seconds, gpu_memory, learning_rate)
        append_row(metrics_path, losses + detection_values + runtime)
        append_row(comparison_path, (epoch,) + detection_values + runtime)
        if dendritic_path is not None:
            for row in dendritic_statistics(model, epoch):
                append_row(dendritic_path, row)

        if detection is None:
            continue
        print(
            f"epoch {epoch:03d}/{args.epochs} train={train_metrics['loss']:.4f} "
            f"val={val_metrics['loss']:.4f} P={detection['precision']:.4f} "
            f"R={detection['recall']:.4f} mAP50={detection['map50']:.4f} "
            f"mAP50-95={detection['map50_95']:.4f}"
        )
        checkpoint = {
            "model": model.state_dict(),
            "model_type": model_name,
            "names": config["names"],
            "width": args.width,
            "branches": args.branches if args.variant == "dendritic" else None,
            "replace_layers": args.replace_layers if args.variant == "dendritic" else 0,
            "img_size": args.img_size,
            "epoch": epoch,
            "validation_metrics": detection,
        }
        torch.save(checkpoint, output / "last.pt")
        if float(detection["map50_95"]) > best_score:
            best_score = float(detection["map50_95"])
            best_epoch = epoch
            best_validation = detection
            torch.save(checkpoint, output / "best.pt")

    if not (output / "best.pt").exists():
        raise RuntimeError("没有生成验证集检查点，请检查 --eval-interval 和 --epochs")
    best = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    test_metrics = evaluate_detection(
        model, test_loader, len(config["names"]), device, config["names"]
    )
    summary = {
        **protocol,
        "best_epoch": best_epoch,
        "best_validation": best_validation,
        "test": test_metrics,
    }
    (output / "test_metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(test_metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

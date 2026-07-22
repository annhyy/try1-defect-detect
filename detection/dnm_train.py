"""使用 APSPC 的 YOLO 框标注训练现有 DNM 检测器。

本入口恢复历史单尺度 DNM 目标检测任务，输出目标置信度、边框和类别，
不再把检测数据转换为整图分类问题。
"""
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
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alfoil_dnm.data import YoloDefectDataset, collate, load_data_yaml
from alfoil_dnm.loss import detector_loss
from alfoil_dnm.metrics import evaluate_detection
from alfoil_dnm.model_variants import MODEL_VARIANTS, build_detector


CONTROLLED_DATA = ROOT / "datasets" / "apspc_yolo_letterbox640" / "data.yaml"
METRIC_COLUMNS = (
    "epoch", "train_total", "train_obj", "train_box", "train_cls",
    "val_total", "val_obj", "val_box", "val_cls",
    "precision", "recall", "map50", "map50_95",
    "epoch_seconds", "elapsed_seconds", "gpu_memory_mb", "learning_rate",
)
COMPARISON_COLUMNS = (
    "epoch", "precision", "recall", "map50", "map50_95",
    "epoch_seconds", "elapsed_seconds", "gpu_memory_mb", "learning_rate",
)


def set_seed(seed: int) -> None:
    """固定 Python、NumPy 和 PyTorch 随机源。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def make_loader(dataset, batch_size: int, workers: int, shuffle: bool, device: torch.device, seed: int):
    """创建使用固定随机种子的检测数据加载器。"""
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
        pin_memory=device.type == "cuda", persistent_workers=workers > 0,
        collate_fn=collate, generator=generator,
    )


def run_epoch(model, loader, optimizer, num_classes: int, device: torch.device, show_progress: bool, label: str):
    """运行一个训练或验证轮次，并汇总各项检测损失。"""
    model.train(optimizer is not None)
    totals = {"loss": 0.0, "obj": 0.0, "box": 0.0, "cls": 0.0}
    batches = tqdm(loader, desc=label, leave=False, disable=not show_progress, dynamic_ncols=True)
    for images, targets, _ in batches:
        images = images.to(device, non_blocking=device.type == "cuda")
        with torch.set_grad_enabled(optimizer is not None):
            prediction = model(images)
            loss, details = detector_loss(prediction, targets, num_classes)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        for key in totals:
            totals[key] += float(details[key]) * images.shape[0]
        batches.set_postfix(loss=f"{float(details['loss']):.4f}")
    return {key: value / max(len(loader.dataset), 1) for key, value in totals.items()}


def _write_row(path: Path, row: tuple) -> None:
    """向逐轮 CSV 追加一行。"""
    with path.open("a", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow(row)


def main(default_variant: str = "v1", default_out: str | Path | None = None, default_branch_features: int = 4) -> None:
    """运行统一的 APSPC DNM 检测训练协议。"""
    if default_variant not in MODEL_VARIANTS:
        raise ValueError(f"未知 DNM 版本：{default_variant}")
    parser = argparse.ArgumentParser(description="APSPC DNM 目标检测训练")
    parser.add_argument("--variant", choices=MODEL_VARIANTS, default=default_variant)
    parser.add_argument("--data", default=str(CONTROLLED_DATA), help="包含 train/val/test 的 YOLO data.yaml")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--branches", type=int, default=4)
    parser.add_argument("--branch-features", type=int, default=default_branch_features)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--out", default=str(default_out or ROOT / "run2" / "controlled" / "dnm_v1"))
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    if args.eval_interval < 1:
        raise ValueError("--eval-interval 必须大于等于 1")

    set_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cfg = load_data_yaml(args.data)
    cached_size = cfg.get("letterbox_size")
    if cached_size is not None and int(cached_size) != args.img_size:
        raise ValueError(
            f"数据已预先 letterbox 为 {cached_size}，不能再按 {args.img_size} 二次缩放。"
        )
    train_set = YoloDefectDataset(cfg, "train", args.img_size)
    val_set = YoloDefectDataset(cfg, "val", args.img_size)
    test_set = YoloDefectDataset(cfg, "test", args.img_size)
    train_loader = make_loader(train_set, args.batch_size, args.workers, True, device, args.seed)
    val_loader = make_loader(val_set, args.batch_size, args.workers, False, device, args.seed)
    test_loader = make_loader(test_set, args.batch_size, args.workers, False, device, args.seed)
    model = build_detector(args.variant, len(cfg["names"]), args.width, args.branches, args.branch_features).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0.0)
    output = Path(args.out).resolve()
    output.mkdir(parents=True, exist_ok=True)
    metrics_path, comparison_path = output / "metrics.csv", output / "comparison_metrics.csv"
    metrics_path.write_text(",".join(METRIC_COLUMNS) + "\n", encoding="utf-8")
    comparison_path.write_text(",".join(COMPARISON_COLUMNS) + "\n", encoding="utf-8")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    protocol = {
        "protocol": "apspc_detection_dnm_v1",
        "task": "object_detection",
        "model": args.variant,
        "data": str(Path(args.data).resolve()),
        "class_names": cfg["names"],
        "epochs": args.epochs, "img_size": args.img_size, "batch_size": args.batch_size,
        "seed": args.seed, "pretrained": False, "augmentation": "none",
        "optimizer": "AdamW", "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay, "scheduler": "CosineAnnealingLR(eta_min=0)",
        "amp": False, "parameters": parameter_count,
        "width": args.width, "branches": args.branches,
        "branch_features": args.branch_features,
        "aggregation": {
            "v1": "direct_product",
            "v2a": "log_domain_exact_product",
            "v2b": "log_domain_geometric_mean",
            "conv": "parameter_matched_convolution",
        }[args.variant],
        "metric_selection": "validation_map50_95",
    }
    (output / "experiment_config.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"任务：APSPC 目标检测；模型：DNM-{args.variant}；输入：{args.img_size}×{args.img_size}")
    print(f"设备：{device}；参数量：{parameter_count:,}；输出：{output}")

    best_score, best_epoch, best_validation = float("-inf"), 0, None
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        epoch_start = time.perf_counter()
        lr = optimizer.param_groups[0]["lr"]
        train = run_epoch(model, train_loader, optimizer, len(cfg["names"]), device, not args.no_progress, f"train {epoch:03d}/{args.epochs}")
        val = run_epoch(model, val_loader, None, len(cfg["names"]), device, not args.no_progress, f"val {epoch:03d}/{args.epochs}")
        detection = evaluate_detection(model, val_loader, len(cfg["names"]), device, cfg["names"]) if epoch % args.eval_interval == 0 or epoch == args.epochs else None
        scheduler.step()
        gpu_memory = torch.cuda.max_memory_allocated(device) / 1024 ** 2 if device.type == "cuda" else 0.0
        epoch_seconds, elapsed = time.perf_counter() - epoch_start, time.perf_counter() - started
        values = (epoch, train["loss"], train["obj"], train["box"], train["cls"], val["loss"], val["obj"], val["box"], val["cls"])
        metrics = ("", "", "", "") if detection is None else tuple(detection[key] for key in ("precision", "recall", "map50", "map50_95"))
        _write_row(metrics_path, values + metrics + (epoch_seconds, elapsed, gpu_memory, lr))
        _write_row(comparison_path, (epoch,) + metrics + (epoch_seconds, elapsed, gpu_memory, lr))
        if detection is not None:
            print(f"epoch {epoch:03d}/{args.epochs} train={train['loss']:.4f} val={val['loss']:.4f} P={detection['precision']:.4f} R={detection['recall']:.4f} mAP50={detection['map50']:.4f} mAP50-95={detection['map50_95']:.4f}")
            checkpoint = {"model": model.state_dict(), "variant": args.variant, "names": cfg["names"], "width": args.width, "branches": args.branches, "branch_features": args.branch_features, "img_size": args.img_size, "epoch": epoch, "validation_metrics": detection}
            torch.save(checkpoint, output / "last.pt")
            if float(detection["map50_95"]) > best_score:
                best_score, best_epoch, best_validation = float(detection["map50_95"]), epoch, detection
                torch.save(checkpoint, output / "best.pt")

    if not (output / "best.pt").exists():
        raise RuntimeError("没有生成验证集检查点，请检查 --eval-interval 和 --epochs")
    best = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    test_metrics = evaluate_detection(model, test_loader, len(cfg["names"]), device, cfg["names"])
    summary = {**protocol, "best_epoch": best_epoch, "best_validation": best_validation, "test": test_metrics}
    (output / "test_metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(test_metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

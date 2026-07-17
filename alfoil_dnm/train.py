"""受控对照协议下的树突缺陷检测训练入口。"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

try:  # 同时兼容 ``python -m alfoil_dnm.train`` 与 IDE 直接运行。
    from .data import YoloDefectDataset, collate, load_data_yaml
    from .loss import detector_loss
    from .metrics import evaluate_detection
    from .model import DendriticDetector
except ImportError:
    from data import YoloDefectDataset, collate, load_data_yaml
    from loss import detector_loss
    from metrics import evaluate_detection
    from model import DendriticDetector


ROOT = Path(__file__).resolve().parents[1]
CONTROLLED_DATA = ROOT / "datasets" / "apspc_yolo_letterbox640" / "data.yaml"
CSV_COLUMNS = (
    "epoch", "train_total", "train_obj", "train_box", "train_cls",
    "val_total", "val_obj", "val_box", "val_cls",
    "precision", "recall", "map50", "map50_95", "f1",
    "epoch_seconds", "elapsed_seconds", "gpu_memory_mb", "learning_rate",
)
COMPARISON_COLUMNS = (
    "epoch", "precision", "recall", "map50", "map50_95",
    "epoch_seconds", "elapsed_seconds", "gpu_memory_mb", "learning_rate",
)


def set_seed(seed: int) -> None:
    """固定 Python、NumPy 与 PyTorch 的随机源，使受控实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def run_epoch(model, loader, optimizer, num_classes, device, description: str, show_progress: bool):
    """运行一个训练或验证 epoch，返回模型内部 loss 用于观察收敛。"""
    model.train(optimizer is not None)
    totals = {"loss": 0.0, "obj": 0.0, "box": 0.0, "cls": 0.0}
    batches = tqdm(loader, desc=description, leave=False, dynamic_ncols=True, disable=not show_progress)
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
    return {key: value / len(loader.dataset) for key, value in totals.items()}


def build_loader(dataset, batch_size: int, workers: int, shuffle: bool, device: torch.device, seed: int) -> DataLoader:
    """建立固定随机种子的 DataLoader；不做图像增强，保持三模型输入一致。"""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        collate_fn=collate,
        generator=generator,
    )


def metric_row(metrics: dict | None) -> tuple[float | str, float | str, float | str, float | str, float | str]:
    """将评估字典转换为 CSV/终端统一字段。"""
    if metrics is None:
        return "", "", "", "", ""
    precision, recall = float(metrics["precision"]), float(metrics["recall"])
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return precision, recall, float(metrics["map50"]), float(metrics["map50_95"]), f1


def main() -> None:
    parser = argparse.ArgumentParser(description="树突检测器：受控 YOLO 对照训练")
    parser.add_argument("--data", default=str(CONTROLLED_DATA), help="三模型共享的 YOLO data.yaml")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--branches", type=int, default=4)
    parser.add_argument("--branch-features", type=int, default=4, help="每个树突分支的乘性突触项数")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--out", default=str(ROOT / "runs" / "controlled" / "dnm"))
    parser.add_argument("--eval-interval", type=int, default=1, help="每隔多少个 epoch 计算一次标准检测指标")
    parser.add_argument("--no-progress", action="store_true", help="关闭每个 batch 的 tqdm 进度条")
    args = parser.parse_args()
    if args.eval_interval < 1:
        raise ValueError("--eval-interval 必须大于等于 1")

    set_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cfg = load_data_yaml(args.data)
    train_set = YoloDefectDataset(cfg, "train", args.img_size)
    val_set = YoloDefectDataset(cfg, "val", args.img_size)
    test_set = YoloDefectDataset(cfg, "test", args.img_size)
    train_loader = build_loader(train_set, args.batch_size, args.workers, True, device, args.seed)
    val_loader = build_loader(val_set, args.batch_size, args.workers, False, device, args.seed)
    test_loader = build_loader(test_set, args.batch_size, args.workers, False, device, args.seed)

    model = DendriticDetector(len(cfg["names"]), args.width, args.branches, args.branch_features).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0.0)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    metrics_path = out / "metrics.csv"
    comparison_path = out / "comparison_metrics.csv"
    # 本训练器不支持断点续训；每次启动均重写 CSV，避免不同运行的指标混在一起。
    with metrics_path.open("w", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow(CSV_COLUMNS)
    with comparison_path.open("w", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow(COMPARISON_COLUMNS)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    protocol = {
        "protocol": "controlled_scratch_v1",
        "model": "dendritic_detector",
        "data": str(Path(args.data).resolve()),
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
        "gradient_accumulation": 1,
        "parameters": parameter_count,
    }
    (out / "experiment_config.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8")
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print(f"受控协议：data={Path(args.data).resolve()}；预训练=False；增强=none；AdamW；AMP=False")
    print(f"设备：{device_name}；模型参数量：{parameter_count:,}；输出目录：{out}")

    best_score, best_epoch, best_validation = float("-inf"), 0, None
    training_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        epoch_start = time.perf_counter()
        train_metrics = run_epoch(model, train_loader, optimizer, len(cfg["names"]), device, f"训练 {epoch:03d}/{args.epochs}", not args.no_progress)
        val_metrics = run_epoch(model, val_loader, None, len(cfg["names"]), device, f"验证 {epoch:03d}/{args.epochs}", not args.no_progress)
        detection_metrics = None
        if epoch % args.eval_interval == 0 or epoch == args.epochs:
            detection_metrics = evaluate_detection(model, val_loader, len(cfg["names"]), device, cfg["names"])
        scheduler.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            gpu_memory_mb = torch.cuda.max_memory_allocated(device) / 1024 ** 2
        else:
            gpu_memory_mb = 0.0
        epoch_seconds = time.perf_counter() - epoch_start
        elapsed_seconds = time.perf_counter() - training_start
        precision, recall, map50, map50_95, f1 = metric_row(detection_metrics)
        print(
            f"epoch {epoch:03d}/{args.epochs} "
            f"train(total={train_metrics['loss']:.4f}, obj={train_metrics['obj']:.4f}, box={train_metrics['box']:.4f}, cls={train_metrics['cls']:.4f}) "
            f"val(total={val_metrics['loss']:.4f}, obj={val_metrics['obj']:.4f}, box={val_metrics['box']:.4f}, cls={val_metrics['cls']:.4f}) "
            f"time={epoch_seconds:.1f}s gpu_mem={gpu_memory_mb:.0f}MB"
        )
        if detection_metrics is not None:
            print(
                f"{'Class':>20} {'Images':>8} {'Instances':>10} {'Box(P':>10} {'R':>8} {'mAP50':>9} {'mAP50-95)':>11}\n"
                f"{'all':>20} {detection_metrics['images']:>8} {detection_metrics['instances']:>10} "
                f"{precision:>10.4f} {recall:>8.4f} {map50:>9.4f} {map50_95:>11.4f}"
            )
        with metrics_path.open("a", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow((
                epoch, train_metrics["loss"], train_metrics["obj"], train_metrics["box"], train_metrics["cls"],
                val_metrics["loss"], val_metrics["obj"], val_metrics["box"], val_metrics["cls"],
                precision, recall, map50, map50_95, f1,
                epoch_seconds, elapsed_seconds, gpu_memory_mb, optimizer.param_groups[0]["lr"],
            ))
        with comparison_path.open("a", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow((
                epoch, precision, recall, map50, map50_95,
                epoch_seconds, elapsed_seconds, gpu_memory_mb, optimizer.param_groups[0]["lr"],
            ))
        checkpoint = {
            "model": model.state_dict(), "names": cfg["names"], "width": args.width,
            "branches": args.branches, "branch_features": args.branch_features, "img_size": args.img_size,
            "epoch": epoch, "validation_metrics": detection_metrics,
        }
        torch.save(checkpoint, out / "last.pt")
        if detection_metrics is not None and float(detection_metrics["map50_95"]) > best_score:
            best_score, best_epoch, best_validation = float(detection_metrics["map50_95"]), epoch, detection_metrics
            torch.save(checkpoint, out / "best.pt")

    best_checkpoint = torch.load(out / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model"])
    test_metrics = evaluate_detection(model, test_loader, len(cfg["names"]), device, cfg["names"])
    summary = {
        **protocol,
        "best_epoch": best_epoch,
        "best_validation": best_validation,
        "test": test_metrics,
    }
    (out / "test_metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("测试集指标：")
    print(json.dumps(test_metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

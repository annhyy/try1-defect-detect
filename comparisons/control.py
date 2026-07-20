"""表面缺陷分类对照协议与 Ultralytics 日志标准化工具。"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


COMPARISON_COLUMNS = (
    "epoch",
    "train_loss", "train_accuracy", "train_macro_precision", "train_macro_recall", "train_macro_f1",
    "val_loss", "val_accuracy", "val_macro_precision", "val_macro_recall", "val_macro_f1",
    "epoch_seconds", "elapsed_seconds", "gpu_memory_mb", "learning_rate",
)


def controlled_classification_options(epochs: int, augmentation: bool) -> dict[str, Any]:
    """返回 YOLO11/26 分类共用的优化和轻量增强设置。"""
    options: dict[str, Any] = {
        "optimizer": "AdamW",
        "lr0": 1e-3,
        "lrf": 0.0,
        "cos_lr": True,
        "weight_decay": 1e-4,
        "warmup_epochs": 0.0,
        "nbs": 64,
        "workers": 2,
        "amp": False,
        "deterministic": True,
        "patience": epochs,
        "erasing": 0.0,
        # 禁用 Ultralytics 额外的自动增强，避免与树突训练器出现隐藏策略差异。
        "auto_augment": None,
    }
    if augmentation:
        options.update({
            "hsv_h": 0.0,
            "hsv_s": 0.0,
            "hsv_v": 0.1,
            "degrees": 10.0,
            "translate": 0.05,
            "scale": 0.1,
            "shear": 0.0,
            "perspective": 0.0,
            "flipud": 0.5,
            "fliplr": 0.5,
        })
    else:
        options.update({
            "hsv_h": 0.0,
            "hsv_s": 0.0,
            "hsv_v": 0.0,
            "degrees": 0.0,
            "translate": 0.0,
            "scale": 0.0,
            "shear": 0.0,
            "perspective": 0.0,
            "flipud": 0.0,
            "fliplr": 0.0,
        })
    return options


def _number(row: dict[str, str], *names: str) -> float | str:
    for name in names:
        value = row.get(name, "")
        if value:
            return float(value)
    return ""


def write_protocol(output: Path, protocol: dict[str, Any]) -> None:
    (output / "experiment_config.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def standardize_yolo_classification_metrics(
    results_csv: Path,
    output_csv: Path,
    gpu_memory_mb_by_epoch: dict[int, float] | None = None,
    validation_metrics_by_epoch: dict[int, dict[str, float]] | None = None,
    epoch_seconds_by_epoch: dict[int, float] | None = None,
) -> None:
    """把 YOLO 分类逐轮日志转为与树突训练器一致的列。

    Ultralytics 原生日志只提供 Top-1 Accuracy；本项目在每轮结束后使用共用
    验证器补算 Accuracy、Macro-Precision、Macro-Recall 和 Macro-F1。
    """
    with results_csv.open("r", newline="", encoding="utf-8-sig") as file:
        rows = [
            {key.strip(): value.strip() for key, value in row.items() if key}
            for row in csv.DictReader(file)
        ]
    raw_epochs = [int(float(row["epoch"])) for row in rows if row.get("epoch", "")]
    epoch_offset = 1 if raw_epochs and raw_epochs[0] == 0 else 0
    previous_elapsed = 0.0
    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(COMPARISON_COLUMNS)
        for row in rows:
            raw_epoch = _number(row, "epoch")
            epoch = int(raw_epoch) + epoch_offset if raw_epoch != "" else ""
            elapsed_value = _number(row, "time")
            native_elapsed = float(elapsed_value) if elapsed_value != "" else previous_elapsed
            if epoch in (epoch_seconds_by_epoch or {}):
                epoch_seconds = epoch_seconds_by_epoch[epoch]
                elapsed = previous_elapsed + epoch_seconds
            else:
                elapsed = native_elapsed
                epoch_seconds = elapsed - previous_elapsed
            previous_elapsed = elapsed
            accuracy = _number(
                row,
                "metrics/accuracy_top1",
                "metrics/accuracy_top1(B)",
                "metrics/accuracy",
            )
            unified = (validation_metrics_by_epoch or {}).get(epoch, {})
            if unified:
                accuracy = unified["accuracy"]
            writer.writerow((
                epoch,
                _number(row, "train/loss"), "", "", "", "",
                unified.get("loss", _number(row, "val/loss")),
                accuracy,
                unified.get("macro_precision", ""),
                unified.get("macro_recall", ""),
                unified.get("macro_f1", ""),
                epoch_seconds,
                elapsed,
                (gpu_memory_mb_by_epoch or {}).get(epoch, ""),
                _number(row, "lr/pg0"),
            ))

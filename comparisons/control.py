"""三模型共用的受控对照协议与 YOLO 日志标准化工具。"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


COMPARISON_COLUMNS = (
    "epoch", "precision", "recall", "map50", "map50_95",
    "epoch_seconds", "elapsed_seconds", "gpu_memory_mb", "learning_rate",
)


def controlled_yolo_options(epochs: int) -> dict[str, Any]:
    """返回与树突模型对齐的 YOLO 训练设置。

    模型骨干与检测损失本身是待比较对象，不能强行相同；其余数据、训练预算、
    优化器、增强、AMP 和名义批量均固定，避免额外变量干扰。
    """
    return {
        "optimizer": "AdamW",
        "lr0": 2e-3,
        "lrf": 0.0,
        "cos_lr": True,
        "weight_decay": 1e-4,
        "warmup_epochs": 0.0,
        "nbs": 8,
        "workers": 2,
        "amp": False,
        "deterministic": True,
        "patience": epochs,
        # 关闭所有几何、颜色与混合增强，使输入样本与树突模型完全一致。
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
        "mosaic": 0.0,
        "close_mosaic": 0,
        "mixup": 0.0,
        "cutmix": 0.0,
        "copy_paste": 0.0,
        "erasing": 0.0,
    }


def _number(row: dict[str, str], *names: str) -> float | str:
    for name in names:
        value = row.get(name, "")
        if value:
            return float(value)
    return ""


def write_protocol(output: Path, protocol: dict[str, Any]) -> None:
    """将实际训练协议和模型初始化方式保存到每个实验输出目录。"""
    (output / "experiment_config.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8")


def standardize_yolo_metrics(
    results_csv: Path,
    output_csv: Path,
    gpu_memory_mb_by_epoch: dict[int, float] | None = None,
) -> None:
    """将 Ultralytics 的 ``results.csv`` 转换为三模型共用指标列。"""
    with results_csv.open("r", newline="", encoding="utf-8-sig") as file:
        rows = [{key.strip(): value.strip() for key, value in row.items() if key} for row in csv.DictReader(file)]
    previous_elapsed = 0.0
    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(COMPARISON_COLUMNS)
        for row in rows:
            elapsed = _number(row, "time")
            elapsed = float(elapsed) if elapsed != "" else previous_elapsed
            epoch_seconds = elapsed - previous_elapsed
            previous_elapsed = elapsed
            epoch = _number(row, "epoch")
            # Ultralytics CSV 从 0 开始计 epoch；树突日志从 1 开始，统一为 1 开始。
            epoch = int(epoch) + 1 if epoch != "" else ""
            writer.writerow((
                epoch,
                _number(row, "metrics/precision(B)", "metrics/precision"),
                _number(row, "metrics/recall(B)", "metrics/recall"),
                _number(row, "metrics/mAP50(B)", "metrics/mAP50"),
                _number(row, "metrics/mAP50-95(B)", "metrics/mAP50-95"),
                epoch_seconds,
                elapsed,
                (gpu_memory_mb_by_epoch or {}).get(epoch, ""),
                _number(row, "lr/pg0"),
            ))

"""分类模型共用的 Accuracy、Macro-F1、混淆矩阵和推理测速。"""
from __future__ import annotations

import copy
import csv
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import Tensor, nn


def extract_logits(output) -> Tensor:
    """兼容普通 PyTorch 模型与 Ultralytics 分类模型的前向输出。"""
    if isinstance(output, Tensor):
        return output
    if isinstance(output, (tuple, list)):
        # Ultralytics 分类模型在 eval 模式返回 (probabilities, logits)。
        tensors = [item for item in output if isinstance(item, Tensor)]
        if not tensors:
            raise TypeError("模型输出中没有 Tensor")
        return tensors[-1]
    if isinstance(output, dict):
        for key in ("logits", "pred", "output"):
            if isinstance(output.get(key), Tensor):
                return output[key]
    raise TypeError(f"无法提取分类 logits：{type(output)!r}")


def metrics_from_predictions(targets: list[int], predictions: list[int], class_names: list[str]) -> dict:
    """不依赖 sklearn 计算宏平均分类指标。"""
    classes = len(class_names)
    matrix = np.zeros((classes, classes), dtype=np.int64)
    for target, prediction in zip(targets, predictions):
        matrix[target, prediction] += 1
    total = int(matrix.sum())
    accuracy = float(np.trace(matrix) / total) if total else 0.0
    per_class = []
    for index, name in enumerate(class_names):
        true_positive = int(matrix[index, index])
        false_positive = int(matrix[:, index].sum() - true_positive)
        false_negative = int(matrix[index, :].sum() - true_positive)
        support = int(matrix[index, :].sum())
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        per_class.append({
            "class": name,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        })
    return {
        "accuracy": accuracy,
        "macro_precision": float(np.mean([row["precision"] for row in per_class])),
        "macro_recall": float(np.mean([row["recall"] for row in per_class])),
        "macro_f1": float(np.mean([row["f1"] for row in per_class])),
        "samples": total,
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
    }


@torch.inference_mode()
def evaluate_classifier(
    model: nn.Module,
    loader,
    device: torch.device,
    class_names: list[str],
    criterion: nn.Module | None = None,
    output_adapter: Callable = extract_logits,
) -> dict:
    """在给定集合上计算平均 loss 与全部分类指标。"""
    model.eval()
    total_loss = 0.0
    targets: list[int] = []
    predictions: list[int] = []
    for images, labels in loader:
        images = images.to(device, non_blocking=device.type == "cuda")
        labels = labels.to(device, non_blocking=device.type == "cuda")
        logits = output_adapter(model(images))
        if criterion is not None:
            total_loss += float(criterion(logits, labels)) * images.shape[0]
        targets.extend(labels.cpu().tolist())
        predictions.extend(logits.argmax(dim=1).cpu().tolist())
    metrics = metrics_from_predictions(targets, predictions, class_names)
    metrics["loss"] = total_loss / max(len(loader.dataset), 1) if criterion is not None else None
    metrics["targets"] = targets
    metrics["predictions"] = predictions
    return metrics


def public_metrics(metrics: dict) -> dict:
    """去除逐样本临时数组，得到可写入 JSON 的评估结果。"""
    return {key: value for key, value in metrics.items() if key not in {"targets", "predictions"}}


def save_confusion_matrix(path: str | Path, metrics: dict, class_names: list[str]) -> None:
    """将混淆矩阵保存为带行列类别名的 CSV。"""
    with Path(path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("true\\pred", *class_names))
        for name, row in zip(class_names, metrics["confusion_matrix"]):
            writer.writerow((name, *row))


def save_predictions(path: str | Path, dataset, metrics: dict) -> None:
    """保存测试集逐图片预测，便于定位错分样本。"""
    with Path(path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow((
            "path", "true_id", "true_class", "predicted_id", "predicted_class", "correct"
        ))
        for (image_path, _), target, prediction in zip(
            dataset.samples, metrics["targets"], metrics["predictions"]
        ):
            writer.writerow((
                image_path,
                target,
                dataset.classes[target],
                prediction,
                dataset.classes[prediction],
                int(target == prediction),
            ))


@torch.inference_mode()
def _benchmark_on_device(
    model: nn.Module,
    device: torch.device,
    image_size: int,
    warmup: int,
    iterations: int,
    output_adapter: Callable,
) -> dict:
    model = model.to(device).eval()
    sample = torch.zeros(1, 3, image_size, image_size, device=device)
    for _ in range(warmup):
        output_adapter(model(sample))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    timings = []
    for _ in range(iterations):
        start = time.perf_counter()
        output_adapter(model(sample))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timings.append((time.perf_counter() - start) * 1000)
    values = np.asarray(timings)
    return {
        "device": str(device),
        "warmup": warmup,
        "iterations": iterations,
        "mean_ms": float(values.mean()),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "images_per_second": float(1000.0 / values.mean()),
    }


def benchmark_classifier(
    model: nn.Module,
    training_device: torch.device,
    image_size: int,
    output_adapter: Callable = extract_logits,
    warmup: int = 30,
    iterations: int = 200,
) -> dict:
    """使用 batch=1 的纯模型前向，分别测试训练GPU与CPU。"""
    results = {}
    if training_device.type == "cuda":
        results["gpu"] = _benchmark_on_device(
            model, training_device, image_size, warmup, iterations, output_adapter
        )
    cpu_model = copy.deepcopy(model).cpu()
    results["cpu"] = _benchmark_on_device(
        cpu_model, torch.device("cpu"), image_size, warmup, iterations, output_adapter
    )
    return results

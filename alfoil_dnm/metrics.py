"""树突检测器的标准目标检测指标。

实现与常见 COCO/YOLO 汇总口径一致的 Precision、Recall、mAP@0.5 和
mAP@0.5:0.95。损失函数只能在同一模型内观察收敛，本模块的指标才用于
与 YOLO11/YOLO26 横向比较。
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch
from torch import Tensor


IOU_THRESHOLDS = tuple(round(0.5 + step * 0.05, 2) for step in range(10))


def box_iou(box1: Tensor, box2: Tensor) -> Tensor:
    """计算归一化 xyxy 格式边框的两两 IoU。"""
    top_left = torch.maximum(box1[:, None, :2], box2[None, :, :2])
    bottom_right = torch.minimum(box1[:, None, 2:], box2[None, :, 2:])
    intersection = (bottom_right - top_left).clamp_min(0).prod(dim=2)
    area1 = (box1[:, 2] - box1[:, 0]).clamp_min(0) * (box1[:, 3] - box1[:, 1]).clamp_min(0)
    area2 = (box2[:, 2] - box2[:, 0]).clamp_min(0) * (box2[:, 3] - box2[:, 1]).clamp_min(0)
    return intersection / (area1[:, None] + area2[None, :] - intersection).clamp_min(1e-6)


def nms(boxes: Tensor, scores: Tensor, threshold: float = 0.5) -> Tensor:
    """对同一类别候选框执行贪心非极大值抑制。"""
    keep, order = [], scores.argsort(descending=True)
    while order.numel():
        index = order[0]
        keep.append(index)
        if order.numel() == 1:
            break
        remaining = order[1:]
        iou = box_iou(boxes[index:index + 1], boxes[remaining]).squeeze(0)
        order = remaining[iou < threshold]
    return torch.stack(keep) if keep else torch.empty(0, dtype=torch.long, device=boxes.device)


def decode_predictions(prediction: Tensor, confidence: float = 0.001, max_detections: int = 300):
    """将检测头输出解码为每张图的 ``[x1,y1,x2,y2,score,class]`` 候选框。

    评估时保留低置信度候选框，让 AP 曲线自行扫描置信度阈值；这与固定阈值
    后再计算 mAP 的做法不同，能够避免低估模型的最佳 Precision/Recall。
    """
    batch, _, height, width = prediction.shape
    objectness = prediction[:, 0].sigmoid()
    box = prediction[:, 1:5].sigmoid()
    class_prob = prediction[:, 5:].sigmoid()
    decoded = []
    for batch_id in range(batch):
        class_score, category = (objectness[batch_id].unsqueeze(0) * class_prob[batch_id]).max(dim=0)
        ys, xs = (class_score >= confidence).nonzero(as_tuple=True)
        if not ys.numel():
            decoded.append(torch.empty((0, 6), device=prediction.device))
            continue
        scores = class_score[ys, xs]
        if scores.numel() > max_detections:
            scores, selected = scores.topk(max_detections)
            ys, xs = ys[selected], xs[selected]
        cx = (xs + box[batch_id, 0, ys, xs]) / width
        cy = (ys + box[batch_id, 1, ys, xs]) / height
        bw, bh = box[batch_id, 2, ys, xs], box[batch_id, 3, ys, xs]
        boxes = torch.stack((cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2), dim=1).clamp(0, 1)
        classes = category[ys, xs]
        result = []
        for cls in classes.unique():
            mask = classes == cls
            keep = nms(boxes[mask], scores[mask])
            result.append(torch.cat((boxes[mask][keep], scores[mask][keep, None], cls.expand(keep.numel(), 1)), dim=1))
        decoded.append(torch.cat(result) if result else torch.empty((0, 6), device=prediction.device))
    return decoded


def average_precision(recall: Tensor, precision: Tensor) -> Tensor:
    """使用 precision 包络线计算 AP。"""
    recall = torch.cat((torch.tensor([0.0]), recall.cpu(), torch.tensor([1.0])))
    precision = torch.cat((torch.tensor([0.0]), precision.cpu(), torch.tensor([0.0])))
    precision = torch.flip(torch.cummax(torch.flip(precision, dims=[0]), dim=0).values, dims=[0])
    return torch.sum((recall[1:] - recall[:-1]) * precision[1:])


def _match_predictions(predictions: list[tuple[int, float, Tensor]], targets: list[tuple[int, Tensor]], iou_threshold: float):
    """按置信度排序并完成一个类别、一个 IoU 阈值下的一对一匹配。"""
    ordered = sorted(predictions, key=lambda item: item[1], reverse=True)
    targets_by_image: dict[int, list[tuple[int, Tensor]]] = defaultdict(list)
    for index, (image_id, target_box) in enumerate(targets):
        targets_by_image[image_id].append((index, target_box))
    used: set[int] = set()
    true_positive, false_positive = [], []
    for image_id, _, box in ordered:
        candidates = [(index, target_box) for index, target_box in targets_by_image[image_id] if index not in used]
        if not candidates:
            true_positive.append(0.0)
            false_positive.append(1.0)
            continue
        indexes, boxes = zip(*candidates)
        ious = box_iou(box.unsqueeze(0), torch.stack(boxes)).squeeze(0)
        best_iou, best_position = ious.max(dim=0)
        if float(best_iou) >= iou_threshold:
            used.add(indexes[int(best_position)])
            true_positive.append(1.0)
            false_positive.append(0.0)
        else:
            true_positive.append(0.0)
            false_positive.append(1.0)
    return torch.tensor(true_positive), torch.tensor(false_positive)


def _class_statistics(predictions: list[tuple[int, float, Tensor]], targets: list[tuple[int, Tensor]]) -> tuple[float, float, float, float]:
    """返回一个类别的 P、R、AP50 与 10 个 IoU 阈值平均 AP。"""
    if not targets:
        return 0.0, 0.0, 0.0, 0.0
    ap_values = []
    precision50 = recall50 = 0.0
    for threshold in IOU_THRESHOLDS:
        true_positive, false_positive = _match_predictions(predictions, targets, threshold)
        if true_positive.numel() == 0:
            ap_values.append(0.0)
            continue
        cumulative_tp, cumulative_fp = true_positive.cumsum(0), false_positive.cumsum(0)
        recall = cumulative_tp / len(targets)
        precision = cumulative_tp / (cumulative_tp + cumulative_fp).clamp_min(1e-6)
        ap_values.append(float(average_precision(recall, precision)))
        if threshold == 0.5:
            # 与 YOLO 的报告口径相近：在 PR 曲线上选取最佳 F1 对应的 P/R。
            f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-6)
            best = int(f1.argmax())
            precision50, recall50 = float(precision[best]), float(recall[best])
    return precision50, recall50, ap_values[0], sum(ap_values) / len(ap_values)


@torch.no_grad()
def evaluate_detection(model, loader, num_classes: int, device: torch.device, class_names: list[str] | None = None) -> dict[str, Any]:
    """计算可与 YOLO 直接对照的验证或测试集指标。"""
    model.eval()
    predictions: dict[int, list[tuple[int, float, Tensor]]] = defaultdict(list)
    ground_truth: dict[int, list[tuple[int, Tensor]]] = defaultdict(list)
    image_offset = 0
    for images, targets, _ in loader:
        decoded = decode_predictions(model(images.to(device)))
        for local_id, detections in enumerate(decoded):
            for detection in detections.cpu():
                predictions[int(detection[5])].append((image_offset + local_id, float(detection[4]), detection[:4]))
        for local_id, labels in enumerate(targets):
            for label in labels:
                category, cx, cy, width, height = label.tolist()
                ground_truth[int(category)].append((image_offset + local_id, torch.tensor([cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2])))
        image_offset += len(targets)

    per_class, precision, recall, map50, map50_95 = {}, [], [], [], []
    for category in range(num_classes):
        p, r, ap50, ap5095 = _class_statistics(predictions[category], ground_truth[category])
        name = class_names[category] if class_names else str(category)
        per_class[name] = {"instances": len(ground_truth[category]), "precision": p, "recall": r, "ap50": ap50, "ap50_95": ap5095}
        if ground_truth[category]:
            precision.append(p)
            recall.append(r)
            map50.append(ap50)
            map50_95.append(ap5095)
    return {
        "images": image_offset,
        "instances": sum(len(values) for values in ground_truth.values()),
        "precision": sum(precision) / len(precision) if precision else 0.0,
        "recall": sum(recall) / len(recall) if recall else 0.0,
        "map50": sum(map50) / len(map50) if map50 else 0.0,
        "map50_95": sum(map50_95) / len(map50_95) if map50_95 else 0.0,
        "per_class": per_class,
    }


@torch.no_grad()
def evaluate_map50(model, loader, num_classes: int, device: torch.device) -> float:
    """兼容旧调用方式，返回 mAP@0.5。"""
    return float(evaluate_detection(model, loader, num_classes, device)["map50"])

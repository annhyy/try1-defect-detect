"""为树突检测器提供与 YOLO 对比使用的 mAP@0.5 指标。"""
from __future__ import annotations

from collections import defaultdict

import torch
from torch import Tensor


def box_iou(box1: Tensor, box2: Tensor) -> Tensor:
    """计算 xyxy 格式边框的两两 IoU。"""
    top_left = torch.maximum(box1[:, None, :2], box2[None, :, :2])
    bottom_right = torch.minimum(box1[:, None, 2:], box2[None, :, 2:])
    intersection = (bottom_right - top_left).clamp_min(0).prod(dim=2)
    area1 = (box1[:, 2] - box1[:, 0]).clamp_min(0) * (box1[:, 3] - box1[:, 1]).clamp_min(0)
    area2 = (box2[:, 2] - box2[:, 0]).clamp_min(0) * (box2[:, 3] - box2[:, 1]).clamp_min(0)
    return intersection / (area1[:, None] + area2[None, :] - intersection).clamp_min(1e-6)


def nms(boxes: Tensor, scores: Tensor, threshold: float = 0.5) -> Tensor:
    """对单一类别候选框进行贪心非极大值抑制。"""
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


def decode_predictions(prediction: Tensor, confidence: float = 0.05, max_detections: int = 100):
    """将检测头输出解码为每张图的 ``[x1,y1,x2,y2,score,class]`` 列表。"""
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


def targets_to_xyxy(targets: list[Tensor]):
    """将 YOLO 标签 ``[cls,cx,cy,w,h]`` 转为带图片编号的 xyxy 目标框。"""
    converted = []
    for image_id, labels in enumerate(targets):
        if not labels.numel():
            continue
        cls, cx, cy, width, height = labels.T
        boxes = torch.stack((cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2), dim=1)
        image = torch.full((labels.shape[0], 1), image_id, dtype=labels.dtype)
        converted.append(torch.cat((image, boxes, cls[:, None]), dim=1))
    return torch.cat(converted) if converted else torch.empty((0, 6))


def average_precision(recall: Tensor, precision: Tensor) -> Tensor:
    """采用 precision-recall 包络线积分计算 AP。"""
    recall = torch.cat((torch.tensor([0.0]), recall.cpu(), torch.tensor([1.0])))
    precision = torch.cat((torch.tensor([0.0]), precision.cpu(), torch.tensor([0.0])))
    precision = torch.flip(torch.cummax(torch.flip(precision, dims=[0]), dim=0).values, dims=[0])
    return torch.sum((recall[1:] - recall[:-1]) * precision[1:])


@torch.no_grad()
def evaluate_map50(model, loader, num_classes: int, device: torch.device, confidence: float = 0.05) -> float:
    """在验证集计算 mAP@0.5；用于与 YOLO 的 metrics/mAP50 对齐。"""
    model.eval()
    predictions, ground_truth = defaultdict(list), defaultdict(list)
    image_offset = 0
    for images, targets, _ in loader:
        output = model(images.to(device))
        decoded = decode_predictions(output, confidence)
        for local_id, detections in enumerate(decoded):
            for det in detections.cpu():
                predictions[int(det[5])].append((image_offset + local_id, float(det[4]), det[:4]))
        for local_id, labels in enumerate(targets):
            for label in labels:
                cls, cx, cy, width, height = label.tolist()
                ground_truth[int(cls)].append((image_offset + local_id, torch.tensor([cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2])))
        image_offset += len(targets)
    aps = []
    for cls in range(num_classes):
        gt = ground_truth[cls]
        if not gt:
            continue
        used = set()
        ordered = sorted(predictions[cls], key=lambda item: item[1], reverse=True)
        tp, fp = [], []
        for image_id, _, box in ordered:
            candidates = [(index, gt_box) for index, (gt_image, gt_box) in enumerate(gt) if gt_image == image_id and index not in used]
            if candidates:
                indexes, boxes = zip(*candidates)
                iou = box_iou(box.unsqueeze(0), torch.stack(boxes)).max()
                if iou >= 0.5:
                    used.add(indexes[int(box_iou(box.unsqueeze(0), torch.stack(boxes)).argmax())])
                    tp.append(1.0); fp.append(0.0); continue
            tp.append(0.0); fp.append(1.0)
        if not tp:
            aps.append(torch.tensor(0.0)); continue
        tp_tensor, fp_tensor = torch.tensor(tp).cumsum(0), torch.tensor(fp).cumsum(0)
        aps.append(average_precision(tp_tensor / len(gt), tp_tensor / (tp_tensor + fp_tensor).clamp_min(1e-6)))
    return float(torch.stack(aps).mean()) if aps else 0.0

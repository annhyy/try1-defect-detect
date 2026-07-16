from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def build_targets(targets: list[Tensor], grid_h: int, grid_w: int, classes: int, device: torch.device):
    obj = torch.zeros(len(targets), 1, grid_h, grid_w, device=device)
    box = torch.zeros(len(targets), 4, grid_h, grid_w, device=device)
    cls = torch.zeros(len(targets), classes, grid_h, grid_w, device=device)
    positive = torch.zeros_like(obj, dtype=torch.bool)
    for batch_index, labels in enumerate(targets):
        for category, cx, cy, width, height in labels.tolist():
            gx, gy = min(int(cx * grid_w), grid_w - 1), min(int(cy * grid_h), grid_h - 1)
            # one center cell per object: adequate for sparse foil defects.
            if positive[batch_index, 0, gy, gx] and obj[batch_index, 0, gy, gx] > width * height:
                continue
            positive[batch_index, 0, gy, gx] = True
            obj[batch_index, 0, gy, gx] = 1
            box[batch_index, :, gy, gx] = torch.tensor([cx * grid_w - gx, cy * grid_h - gy, width, height], device=device)
            cls[batch_index, int(category), gy, gx] = 1
    return obj, box, cls, positive


def detector_loss(prediction: Tensor, targets: list[Tensor], num_classes: int):
    obj_t, box_t, cls_t, positive = build_targets(targets, prediction.shape[2], prediction.shape[3], num_classes, prediction.device)
    obj_loss = F.binary_cross_entropy_with_logits(prediction[:, :1], obj_t, pos_weight=torch.tensor(4.0, device=prediction.device))
    if positive.any():
        box_loss = F.smooth_l1_loss(torch.sigmoid(prediction[:, 1:5])[positive.expand_as(prediction[:, 1:5])], box_t[positive.expand_as(box_t)])
        cls_loss = F.binary_cross_entropy_with_logits(prediction[:, 5:][positive.expand_as(prediction[:, 5:])], cls_t[positive.expand_as(cls_t)])
    else:
        box_loss, cls_loss = prediction.sum() * 0, prediction.sum() * 0
    total = obj_loss + 5.0 * box_loss + cls_loss
    return total, {"loss": total.detach(), "obj": obj_loss.detach(), "box": box_loss.detach(), "cls": cls_loss.detach()}

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

# 同时兼容 ``python -m alfoil_dnm.infer`` 与 IDE 直接运行 infer.py。
try:
    from .data import load_data_yaml
    from .model_variants import build_detector
except ImportError:
    from data import load_data_yaml
    from model_variants import build_detector


def nms(boxes, scores, threshold=0.45):
    """对同一类别候选框执行贪心非极大值抑制，移除高度重叠的重复框。"""
    keep = []
    order = scores.argsort(descending=True)
    area = (boxes[:, 2] - boxes[:, 0]).clamp_min(0) * (boxes[:, 3] - boxes[:, 1]).clamp_min(0)
    while len(order):
        index = order[0]
        keep.append(index)
        if len(order) == 1:
            break
        rest = order[1:]
        left_top = torch.maximum(boxes[index, :2], boxes[rest, :2])
        right_bottom = torch.minimum(boxes[index, 2:], boxes[rest, 2:])
        intersection = (right_bottom - left_top).clamp_min(0).prod(dim=1)
        iou = intersection / (area[index] + area[rest] - intersection).clamp_min(1e-6)
        order = rest[iou < threshold]
    return torch.stack(keep)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="训练生成的 best.pt 或 last.pt")
    parser.add_argument("--source", required=True, help="待检测的单张图片路径")
    parser.add_argument("--data", required=True, help="与权重类别顺序一致的 data.yaml")
    parser.add_argument("--conf", type=float, default=.35, help="置信度阈值")
    parser.add_argument("--out", default="prediction.jpg", help="带检测框的输出图片路径")
    args = parser.parse_args()
    ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
    names = load_data_yaml(args.data)["names"]
    variant = ckpt.get("variant", "v1")  # 旧权重没有该字段，按 V1 兼容加载。
    model = build_detector(variant, len(names), ckpt["width"], ckpt["branches"], ckpt.get("branch_features", 4))
    model.load_state_dict(ckpt["model"])
    model.eval()
    original = Image.open(args.source).convert("RGB")
    w, h = original.size
    size = ckpt["img_size"]
    # 推理尺寸必须与训练保存的 img_size 一致；随后再映射回原图坐标。
    resized = original.resize((size, size))
    image = torch.from_numpy(np.asarray(resized, dtype=np.float32).transpose(2, 0, 1)).div(255).unsqueeze(0)
    out = model(image)[0]; obj = out[0].sigmoid(); boxes = out[1:5].sigmoid(); classes = out[5:].sigmoid()
    candidates = []
    for y, x in (obj > args.conf).nonzero().tolist():
        score, category = (obj[y, x] * classes[:, y, x]).max(dim=0)
        # 中心点先从特征图坐标还原到输入尺寸，再按原图宽高缩放。
        cx, cy, bw, bh = boxes[:, y, x]
        gx, gy = x + cx, y + cy
        candidates.append(([float((gx * 8 / size - bw / 2) * w), float((gy * 8 / size - bh / 2) * h), float((gx * 8 / size + bw / 2) * w), float((gy * 8 / size + bh / 2) * h)], float(score), int(category)))
    draw = ImageDraw.Draw(original)
    for category in range(len(names)):
        class_candidates = [item for item in candidates if item[2] == category]
        if not class_candidates:
            continue
        class_boxes = torch.tensor([item[0] for item in class_candidates])
        class_scores = torch.tensor([item[1] for item in class_candidates])
        for index in nms(class_boxes, class_scores):
            box, score, _ = class_candidates[int(index)]
            draw.rectangle(box, outline="red", width=2)
            draw.text((box[0], max(0, box[1] - 14)), f"{names[category]} {score:.2f}", fill="red")
    original.save(args.out)
    print(f"模型：{variant}；输出：{Path(args.out).resolve()}")


if __name__ == "__main__":
    main()

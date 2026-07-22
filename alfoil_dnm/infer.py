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


def letterbox_image(image: Image.Image, size: int):
    """将原图等比例缩放并填充到正方形，返回缩放比例和左上填充值。"""
    source_width, source_height = image.size
    scale = min(size / source_width, size / source_height)
    resized_width = round(source_width * scale)
    resized_height = round(source_height * scale)
    pad_x = (size - resized_width) // 2
    pad_y = (size - resized_height) // 2
    resized = image.resize((resized_width, resized_height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    canvas.paste(resized, (pad_x, pad_y))
    return canvas, scale, pad_x, pad_y


def restore_box(box, scale: float, pad_x: int, pad_y: int, width: int, height: int):
    """把 640 letterbox 坐标去除填充并映射回原图像素坐标。"""
    x1, y1, x2, y2 = box
    return [
        max(0.0, min(width, (x1 - pad_x) / scale)),
        max(0.0, min(height, (y1 - pad_y) / scale)),
        max(0.0, min(width, (x2 - pad_x) / scale)),
        max(0.0, min(height, (y2 - pad_y) / scale)),
    ]


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
    # 推理预处理必须与训练缓存一致：等比例缩放、灰边填充，不拉伸缺陷形状。
    resized, scale, pad_x, pad_y = letterbox_image(original, size)
    image = torch.from_numpy(np.asarray(resized, dtype=np.float32).transpose(2, 0, 1)).div(255).unsqueeze(0)
    out = model(image)[0]; obj = out[0].sigmoid(); boxes = out[1:5].sigmoid(); classes = out[5:].sigmoid()
    candidates = []
    grid_height, grid_width = obj.shape
    for y, x in (obj > args.conf).nonzero().tolist():
        score, category = (obj[y, x] * classes[:, y, x]).max(dim=0)
        # 先还原到 letterbox 输入像素，再去除填充并映射回原图。
        cx, cy, bw, bh = boxes[:, y, x]
        center_x = float((x + cx) / grid_width * size)
        center_y = float((y + cy) / grid_height * size)
        box_width = float(bw * size)
        box_height = float(bh * size)
        input_box = [
            center_x - box_width / 2, center_y - box_height / 2,
            center_x + box_width / 2, center_y + box_height / 2,
        ]
        candidates.append((restore_box(input_box, scale, pad_x, pad_y, w, h), float(score), int(category)))
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

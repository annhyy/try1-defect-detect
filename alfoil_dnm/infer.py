from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw

# Allow this script to be launched directly as well as via `python -m`.
try:
    from .data import load_data_yaml
    from .model import DendriticDetector
except ImportError:
    from data import load_data_yaml
    from model import DendriticDetector


def nms(boxes, scores, threshold=0.45):
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
    parser = argparse.ArgumentParser(); parser.add_argument("--weights", required=True); parser.add_argument("--source", required=True); parser.add_argument("--data", required=True); parser.add_argument("--conf", type=float, default=.35); parser.add_argument("--out", default="prediction.jpg")
    args = parser.parse_args(); ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
    names = load_data_yaml(args.data)["names"]
    model = DendriticDetector(len(names), ckpt["width"], ckpt["branches"]); model.load_state_dict(ckpt["model"]); model.eval()
    original = Image.open(args.source).convert("RGB"); w, h = original.size; size = ckpt["img_size"]
    image = torch.tensor(list(original.resize((size, size)).getdata()), dtype=torch.float32).view(size, size, 3).permute(2, 0, 1).div(255).unsqueeze(0)
    out = model(image)[0]; obj = out[0].sigmoid(); boxes = out[1:5].sigmoid(); classes = out[5:].sigmoid()
    candidates = []
    for y, x in (obj > args.conf).nonzero().tolist():
        score, category = (obj[y, x] * classes[:, y, x]).max(dim=0)
        cx, cy, bw, bh = boxes[:, y, x]; gx, gy = x + cx, y + cy
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
            draw.rectangle(box, outline="red", width=2); draw.text((box[0], max(0, box[1]-14)), f"{names[category]} {score:.2f}", fill="red")
    original.save(args.out); print(Path(args.out).resolve())


if __name__ == "__main__":
    main()

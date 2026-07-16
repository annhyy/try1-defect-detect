"""Create a deterministic, labeled synthetic foil-defect dataset for smoke tests."""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import yaml
from PIL import Image, ImageDraw, ImageFilter


NAMES = ["hole", "scratch", "pit", "stain", "insect"]


def make_one(index: int, output: Path, size: int, rng: random.Random) -> None:
    base = Image.new("RGB", (size, size))
    pixels = base.load()
    for y in range(size):
        for x in range(size):
            shade = int(164 + 20 * (x / size) + 10 * (y / size) + rng.uniform(-4, 4))
            pixels[x, y] = (shade, shade + 3, min(255, shade + 8))
    image = base.filter(ImageFilter.GaussianBlur(0.35))
    draw = ImageDraw.Draw(image)
    category = index % len(NAMES)
    cx, cy = rng.randint(42, size - 42), rng.randint(42, size - 42)
    if category == 0:  # hole
        radius = rng.randint(10, 20); box = (cx-radius, cy-radius, cx+radius, cy+radius); draw.ellipse(box, fill=(25, 25, 27))
    elif category == 1:  # scratch
        length, width = rng.randint(45, 75), rng.randint(3, 6); box = (cx-length//2, cy-width//2, cx+length//2, cy+width//2); draw.rounded_rectangle(box, radius=width, fill=(75, 80, 85))
    elif category == 2:  # pit
        radius = rng.randint(10, 18); box = (cx-radius, cy-radius, cx+radius, cy+radius); draw.ellipse(box, fill=(125, 129, 135), outline=(75, 78, 82), width=2)
    elif category == 3:  # stain
        bw, bh = rng.randint(25, 50), rng.randint(18, 38); box = (cx-bw//2, cy-bh//2, cx+bw//2, cy+bh//2); draw.ellipse(box, fill=(115, 105, 80))
    else:  # insect
        radius = rng.randint(7, 11); box = (cx-radius, cy-radius, cx+radius, cy+radius); draw.ellipse(box, fill=(35, 32, 27)); draw.line((cx-radius*2, cy, cx+radius*2, cy), fill=(35, 32, 27), width=2)
    split = "train" if index < 120 else "val"
    stem = f"demo_{index:04d}"
    image.save(output / "images" / split / f"{stem}.jpg", quality=92)
    x1, y1, x2, y2 = box
    line = f"{category} {(x1+x2)/2/size:.6f} {(y1+y2)/2/size:.6f} {(x2-x1)/size:.6f} {(y2-y1)/size:.6f}\n"
    (output / "labels" / split / f"{stem}.txt").write_text(line, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--out", default="demo_alfoil"); parser.add_argument("--size", type=int, default=256); args = parser.parse_args()
    output = Path(args.out).resolve()
    for split in ("train", "val"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)
    for index in range(150):
        make_one(index, output, args.size, rng)
    (output / "data.yaml").write_text(yaml.safe_dump({"path": str(output), "train": "images/train", "val": "images/val", "names": NAMES}, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(output / "data.yaml")


if __name__ == "__main__":
    main()

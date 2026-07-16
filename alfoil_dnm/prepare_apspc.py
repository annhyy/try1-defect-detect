"""Convert locally extracted APSPC Pascal VOC annotations to YOLO detection data."""
from __future__ import annotations

import argparse
import os
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import yaml


CLASSES = [
    "aoxian", "budaodian", "cahua", "jupi", "loudi",
    "pengshang", "qikeng", "tucengkailie", "tufen", "zangdian",
]


def hardlink_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def parse_annotation(path: Path):
    root = ET.parse(path).getroot()
    width = float(root.findtext("size/width"))
    height = float(root.findtext("size/height"))
    labels = []
    for obj in root.findall("object"):
        name = obj.findtext("name")
        if name not in CLASSES:
            raise ValueError(f"Unknown class {name!r} in {path}")
        box = obj.find("bndbox")
        xmin, ymin = float(box.findtext("xmin")), float(box.findtext("ymin"))
        xmax, ymax = float(box.findtext("xmax")), float(box.findtext("ymax"))
        xmin, xmax = max(0.0, min(xmin, width)), max(0.0, min(xmax, width))
        ymin, ymax = max(0.0, min(ymin, height)), max(0.0, min(ymax, height))
        if xmax <= xmin or ymax <= ymin:
            continue
        labels.append((CLASSES.index(name), (xmin + xmax) / (2 * width), (ymin + ymax) / (2 * height), (xmax - xmin) / width, (ymax - ymin) / height))
    return labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="datasets", help="Folder containing APSPC1, APSPC2, and APSPC-Annotations")
    parser.add_argument("--out", default="datasets/apspc_yolo")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    source, output = Path(args.source).resolve(), Path(args.out).resolve()
    xml_dir = source / "APSPC-Annotations" / "Annotations"
    xml_files = sorted(xml_dir.glob("*.xml"))
    if not xml_files:
        raise FileNotFoundError(f"No XML files in {xml_dir}")
    image_paths = list((source / "APSPC1").rglob("*.jpg")) + list((source / "APSPC2").rglob("*.jpg"))
    images = {path.name: path for path in image_paths}
    if len(images) != len(image_paths):
        raise ValueError("Duplicate image file names found; conversion would be ambiguous")
    records = []
    for xml_path in xml_files:
        image_path = images.get(f"{xml_path.stem}.jpg")
        if image_path is None:
            raise FileNotFoundError(f"Image for {xml_path.name} was not found")
        records.append((xml_path, image_path, parse_annotation(xml_path)))
    rng = random.Random(args.seed)
    rng.shuffle(records)
    train_end, val_end = round(len(records) * .7), round(len(records) * .9)
    split_records = {"train": records[:train_end], "val": records[train_end:val_end], "test": records[val_end:]}
    for split, items in split_records.items():
        for xml_path, image_path, labels in items:
            hardlink_or_copy(image_path, output / "images" / split / image_path.name)
            label_path = output / "labels" / split / f"{xml_path.stem}.txt"
            label_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.write_text("".join(f"{category} {cx:.6f} {cy:.6f} {width:.6f} {height:.6f}\n" for category, cx, cy, width, height in labels), encoding="utf-8")
    config = {"path": str(output), "train": "images/train", "val": "images/val", "test": "images/test", "names": CLASSES}
    (output / "data.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"Converted {len(records)} images to {output}")
    for split, items in split_records.items():
        counts = Counter(category for _, _, labels in items for category, *_ in labels)
        print(f"{split}: images={len(items)} boxes={sum(counts.values())} class_boxes={dict(sorted(counts.items()))}")


if __name__ == "__main__":
    main()

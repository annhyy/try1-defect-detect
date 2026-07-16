"""将本地 APSPC 的 Pascal VOC XML 标注转换为 YOLO 检测数据。"""
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
    """优先建立硬链接，失败时复制文件，且不修改原始图片。"""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def parse_annotation(path: Path):
    """读取一个 VOC XML，返回 XML 指定的图片名及其归一化 YOLO 标签。"""
    root = ET.parse(path).getroot()
    filename = root.findtext("filename")
    if not filename:
        raise ValueError(f"XML 缺少 <filename>：{path}")
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
    return filename, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="datasets", help="包含 APSPC1、APSPC2、APSPC-Annotations 的目录")
    parser.add_argument("--out", default="datasets/apspc_yolo", help="YOLO 格式输出目录")
    parser.add_argument("--seed", type=int, default=42, help="训练/验证/测试划分随机种子")
    args = parser.parse_args()
    source, output = Path(args.source).resolve(), Path(args.out).resolve()
    xml_dir = source / "APSPC-Annotations" / "Annotations"
    xml_files = sorted(xml_dir.glob("*.xml"))
    if not xml_files:
        raise FileNotFoundError(f"No XML files in {xml_dir}")
    # 同时兼容 .jpg/.JPG/.png 等扩展名，并用 XML 的 <filename> 精确匹配。
    image_paths = [
        path for folder in (source / "APSPC1", source / "APSPC2")
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    ]
    images = {path.name.lower(): path for path in image_paths}
    if len(images) != len(image_paths):
        raise ValueError("Duplicate image file names found; conversion would be ambiguous")
    records = []
    for xml_path in xml_files:
        image_name, labels = parse_annotation(xml_path)
        image_path = images.get(image_name.lower())
        if image_path is None:
            raise FileNotFoundError(f"找不到 {xml_path.name} 指定的图片：{image_name}")
        records.append((xml_path, image_path, labels))
    rng = random.Random(args.seed)
    # 固定种子保证每次转换产生相同的数据划分，便于实验复现。
    rng.shuffle(records)
    train_end, val_end = round(len(records) * .7), round(len(records) * .9)
    split_records = {"train": records[:train_end], "val": records[train_end:val_end], "test": records[val_end:]}
    for split, items in split_records.items():
        for xml_path, image_path, labels in items:
            hardlink_or_copy(image_path, output / "images" / split / image_path.name)
            label_path = output / "labels" / split / f"{xml_path.stem}.txt"
            label_path.parent.mkdir(parents=True, exist_ok=True)
            # 每行依次为：类别 ID、归一化中心点 x/y、归一化宽/高。
            label_path.write_text("".join(f"{category} {cx:.6f} {cy:.6f} {width:.6f} {height:.6f}\n" for category, cx, cy, width, height in labels), encoding="utf-8")
    config = {"path": str(output), "train": "images/train", "val": "images/val", "test": "images/test", "names": CLASSES}
    (output / "data.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"Converted {len(records)} images to {output}")
    for split, items in split_records.items():
        counts = Counter(category for _, _, labels in items for category, *_ in labels)
        print(f"{split}: images={len(items)} boxes={sum(counts.values())} class_boxes={dict(sorted(counts.items()))}")


if __name__ == "__main__":
    main()

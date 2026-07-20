"""把下载的 NEU YOLO 数据整理为标准图像分类目录。

下载包中的 TXT 是目标检测框标签；图像分类不读取这些 TXT，而是使用
文件名中的主类别（例如 ``crazing_10.jpg`` 属于 ``crazing``）。本脚本
保留原始下载目录不变，在新目录中用硬链接生成 train/val/test 分类数据。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = (
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def class_from_filename(path: Path) -> str:
    """从官方 NEU 文件名中提取图像级主类别。"""
    matches = [name for name in CLASS_NAMES if path.stem.startswith(f"{name}_")]
    if len(matches) != 1:
        raise ValueError(f"无法从文件名确定唯一类别：{path}")
    return matches[0]


def hardlink_or_copy(source: Path, destination: Path) -> None:
    """优先创建硬链接；若文件系统不支持，再复制图像。"""
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def collect_unique_images(source: Path) -> tuple[dict[str, list[Path]], list[dict[str, str]]]:
    """收集图片并去除完全相同的重复文件，避免重复图跨集合泄漏。"""
    candidates = sorted(
        path for path in source.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and path.parent.name == "images"
    )
    if not candidates:
        raise FileNotFoundError(f"没有在 {source} 下找到 images 目录中的图片")

    by_class: dict[str, list[Path]] = defaultdict(list)
    hash_owner: dict[str, tuple[str, Path]] = {}
    excluded: list[dict[str, str]] = []
    for path in candidates:
        class_name = class_from_filename(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        previous = hash_owner.get(digest)
        if previous is not None:
            previous_class, previous_path = previous
            if previous_class != class_name:
                raise ValueError(
                    "发现跨类别的完全重复图片，不能自动决定标签："
                    f"{previous_path} <-> {path}"
                )
            excluded.append({
                "excluded": str(path.resolve()),
                "same_as": str(previous_path.resolve()),
                "class": class_name,
                "sha256": digest,
            })
            continue
        hash_owner[digest] = (class_name, path)
        by_class[class_name].append(path)

    missing = [name for name in CLASS_NAMES if not by_class[name]]
    if missing:
        raise ValueError(f"缺少类别：{missing}")
    return dict(by_class), excluded


def split_images(
    by_class: dict[str, list[Path]], seed: int, val_ratio: float, test_ratio: float
) -> list[tuple[str, str, Path]]:
    """逐类别打乱后分层划分，保证三个集合的类别比例基本一致。"""
    rng = random.Random(seed)
    records: list[tuple[str, str, Path]] = []
    for class_name in CLASS_NAMES:
        images = sorted(by_class[class_name])
        rng.shuffle(images)
        val_count = round(len(images) * val_ratio)
        test_count = round(len(images) * test_ratio)
        train_count = len(images) - val_count - test_count
        if train_count <= 0:
            raise ValueError(f"{class_name} 的训练样本不足")

        boundaries = (
            ("train", images[:train_count]),
            ("val", images[train_count:train_count + val_count]),
            ("test", images[train_count + val_count:]),
        )
        for split, paths in boundaries:
            records.extend((split, class_name, path) for path in paths)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="整理 NEU-CLS 图像分类数据")
    parser.add_argument("--source", default=str(ROOT / "datasets" / "NEU-CLS"))
    parser.add_argument("--out", default=str(ROOT / "datasets" / "neu_cls_classification"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    args = parser.parse_args()

    source = Path(args.source).resolve()
    output = Path(args.out).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"源数据目录不存在：{source}")
    if output.exists():
        raise FileExistsError(f"输出目录已存在，为避免误覆盖请先检查：{output}")
    if args.val_ratio < 0 or args.test_ratio < 0 or args.val_ratio + args.test_ratio >= 1:
        raise ValueError("val-ratio 和 test-ratio 必须非负，且二者之和小于 1")

    by_class, excluded = collect_unique_images(source)
    records = split_images(by_class, args.seed, args.val_ratio, args.test_ratio)
    for split, class_name, source_path in records:
        hardlink_or_copy(source_path, output / split / class_name / source_path.name)

    with (output / "split_manifest.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("split", "class", "filename", "source"))
        for split, class_name, source_path in records:
            writer.writerow((split, class_name, source_path.name, str(source_path.resolve())))

    counts = Counter((split, class_name) for split, class_name, _ in records)
    summary = {
        "source": str(source),
        "output": str(output),
        "task": "classification",
        "class_names": list(CLASS_NAMES),
        "seed": args.seed,
        "split_ratios": {
            "train": 1 - args.val_ratio - args.test_ratio,
            "val": args.val_ratio,
            "test": args.test_ratio,
        },
        "image_count": len(records),
        "excluded_exact_duplicates": excluded,
        "counts": {
            split: {class_name: counts[(split, class_name)] for class_name in CLASS_NAMES}
            for split in ("train", "val", "test")
        },
        "label_rule": "文件名主类别；不读取目标检测 TXT 框标签",
        "resize_rule": "保留 200x200 原图，训练加载器统一缩放到模型输入尺寸",
    }
    (output / "dataset_info.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

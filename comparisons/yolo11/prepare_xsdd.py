"""把原始 X-SDD 整理为 Ultralytics 分类训练需要的目录结构。

原始数据按类别存放在 ``datasets/X-SDD/datas/<类别>/``。本脚本会：

1. 计算每张图片的 SHA-256，排除完全相同的重复图；
2. 在每个类别内部按 70%/15%/15% 划分 train/val/test；
3. 使用硬链接生成分类目录（不重复占用磁盘，失败时退回普通复制）；
4. 保存逐图划分清单和数据统计，便于复现实验并检查数据泄漏。

脚本不会修改或删除原始 X-SDD 文件。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "datasets" / "X-SDD" / "datas"
DEFAULT_OUTPUT = ROOT / "datasets" / "xsdd_yolo11_classification"
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def file_sha256(path: Path) -> str:
    """分块计算文件哈希，避免一次把大文件全部读入内存。"""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def link_or_copy(source: Path, destination: Path) -> str:
    """优先建立硬链接；如果文件系统不支持，则退回普通复制。"""
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def split_count(total: int) -> tuple[int, int, int]:
    """返回约 70%/15%/15% 的三个数量，并保证三者之和等于总数。"""
    train = round(total * 0.70)
    val = round(total * 0.15)
    test = total - train - val
    return train, val, test


def prepare(source: Path, output: Path, seed: int) -> dict:
    source = source.resolve()
    output = output.resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"找不到 X-SDD 类别目录：{source}")
    if output.exists():
        raise FileExistsError(
            f"输出目录已经存在：{output}\n"
            "为防止新旧划分混用，请确认后手动删除该目录，再重新运行。"
        )

    class_dirs = sorted(path for path in source.iterdir() if path.is_dir())
    if not class_dirs:
        raise ValueError(f"目录中没有类别子文件夹：{source}")

    # 同一哈希只允许属于一个类别；若跨类别重复，标签本身存在冲突，应立即停止。
    hash_records: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    for class_dir in class_dirs:
        images = sorted(
            path
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not images:
            raise ValueError(f"类别中没有可识别的图片：{class_dir}")
        for image in images:
            hash_records[file_sha256(image)].append((class_dir.name, image))

    duplicate_groups = []
    excluded_paths: set[Path] = set()
    for digest, records in sorted(hash_records.items()):
        if len(records) == 1:
            continue
        classes = {class_name for class_name, _ in records}
        if len(classes) != 1:
            detail = ", ".join(f"{name}/{path.name}" for name, path in records)
            raise ValueError(f"发现跨类别的完全重复图，标签冲突：{detail}")
        # 按文件名排序后保留第一张，其余只从准备集排除，原始数据完全不动。
        records = sorted(records, key=lambda item: item[1].name.lower())
        kept = records[0][1]
        removed = [path for _, path in records[1:]]
        excluded_paths.update(removed)
        duplicate_groups.append({
            "sha256": digest,
            "class_name": records[0][0],
            "kept": str(kept.relative_to(source)),
            "excluded": [str(path.relative_to(source)) for path in removed],
        })

    manifest_rows: list[dict[str, str]] = []
    split_counts: dict[str, dict[str, int]] = {}
    storage_modes: set[str] = set()
    for class_dir in class_dirs:
        images = sorted(
            path
            for path in class_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in IMAGE_SUFFIXES
            and path not in excluded_paths
        )
        # 每个类别使用独立的、确定性的随机序列。新增其他类别不会改变已有类别划分。
        rng = random.Random(f"X-SDD:{seed}:{class_dir.name}")
        rng.shuffle(images)
        train_count, val_count, test_count = split_count(len(images))
        boundaries = (train_count, train_count + val_count)
        groups = {
            "train": images[: boundaries[0]],
            "val": images[boundaries[0] : boundaries[1]],
            "test": images[boundaries[1] :],
        }
        split_counts[class_dir.name] = {
            split: len(split_images) for split, split_images in groups.items()
        }
        assert len(groups["test"]) == test_count

        for split, split_images in groups.items():
            for image in split_images:
                destination = output / split / class_dir.name / image.name
                storage_modes.add(link_or_copy(image, destination))
                manifest_rows.append({
                    "split": split,
                    "class_name": class_dir.name,
                    "filename": image.name,
                    "source": str(image.relative_to(source)),
                    "sha256": file_sha256(image),
                })

    with (output / "split_manifest.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=("split", "class_name", "filename", "source", "sha256"),
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    totals = {
        split: sum(counts[split] for counts in split_counts.values())
        for split in ("train", "val", "test")
    }
    summary = {
        "dataset": "X-SDD (Xsteel surface defect dataset)",
        "task": "seven-class image classification",
        "source": str(source),
        "output": str(output),
        "seed": seed,
        "split_ratio": {"train": 0.70, "val": 0.15, "test": 0.15},
        "class_names": [path.name for path in class_dirs],
        "raw_images": sum(len(records) for records in hash_records.values()),
        "usable_images": len(manifest_rows),
        "excluded_exact_duplicates": len(excluded_paths),
        "duplicate_groups": duplicate_groups,
        "counts_by_class": split_counts,
        "counts_by_split": totals,
        "storage": sorted(storage_modes),
    }
    (output / "dataset_info.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="准备 X-SDD 的 YOLO11 分类数据")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    summary = prepare(args.source, args.output, args.seed)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

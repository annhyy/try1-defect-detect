"""将 YOLO 数据集缓存为等比例 letterbox 图像，减少训练时的重复解码与缩放。"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml
from PIL import Image


def letterbox(image: Image.Image, size: int, fill: tuple[int, int, int] = (114, 114, 114)):
    """等比例缩放图像，并用灰色填充为 ``size × size``。

    例如 APSPC 的 2560×1920 图像会变为 640×480，再在上下各填充 80 像素。
    返回缓存图像以及将原图像素坐标映射到缓存图的 ``scale, pad_x, pad_y``。
    """
    source_width, source_height = image.size
    scale = min(size / source_width, size / source_height)
    resized_width, resized_height = round(source_width * scale), round(source_height * scale)
    pad_x = (size - resized_width) // 2
    pad_y = (size - resized_height) // 2
    resized = image.resize((resized_width, resized_height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), fill)
    canvas.paste(resized, (pad_x, pad_y))
    return canvas, scale, pad_x, pad_y


def transform_label(line: str, source_width: int, source_height: int, scale: float, pad_x: int, pad_y: int, size: int):
    """将一行 YOLO 归一化标签同步映射到 letterbox 后的正方形坐标系。"""
    category, cx, cy, width, height = line.split()
    cx, cy, width, height = map(float, (cx, cy, width, height))
    # 先恢复原图像素坐标，再应用等比例缩放和填充，最后按 640×640 归一化。
    cx = (cx * source_width * scale + pad_x) / size
    cy = (cy * source_height * scale + pad_y) / size
    width = width * source_width * scale / size
    height = height * source_height * scale / size
    return f"{category} {cx:.6f} {cy:.6f} {width:.6f} {height:.6f}"


def main():
    parser = argparse.ArgumentParser(description="缓存等比例 letterbox 后的 APSPC YOLO 图像")
    parser.add_argument("--source", default="datasets/apspc_yolo", help="原始 YOLO 数据目录")
    parser.add_argument("--out", default="datasets/apspc_yolo_letterbox640", help="缓存数据目录")
    parser.add_argument("--size", type=int, default=640, help="缓存图像边长")
    parser.add_argument("--jpeg-quality", type=int, default=95, help="缓存 JPEG 质量")
    args = parser.parse_args()

    source, output = Path(args.source).resolve(), Path(args.out).resolve()
    source_config = yaml.safe_load((source / "data.yaml").read_text(encoding="utf-8"))
    total = 0
    for split in ("train", "val", "test"):
        image_dir = source / source_config[split]
        label_dir = source / "labels" / split
        images = sorted(path for extension in ("*.jpg", "*.jpeg", "*.png", "*.bmp") for path in image_dir.rglob(extension))
        if not images:
            raise FileNotFoundError(f"未找到 {split} 图像：{image_dir}")
        for index, image_path in enumerate(images, 1):
            target_image = output / "images" / split / image_path.name
            target_label = output / "labels" / split / f"{image_path.stem}.txt"
            target_image.parent.mkdir(parents=True, exist_ok=True)
            target_label.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(image_path) as opened:
                image = opened.convert("RGB")
                cached, scale, pad_x, pad_y = letterbox(image, args.size)
                source_width, source_height = image.size
            # 所有 APSPC 原图为 JPEG；保留扩展名并用高质量 JPEG 缓存，节约读取带宽。
            save_kwargs = {"quality": args.jpeg_quality, "subsampling": 0} if target_image.suffix.lower() in {".jpg", ".jpeg"} else {}
            cached.save(target_image, **save_kwargs)
            source_label = label_dir / f"{image_path.stem}.txt"
            rows = source_label.read_text(encoding="utf-8").splitlines() if source_label.exists() else []
            target_label.write_text(
                "\n".join(transform_label(row, source_width, source_height, scale, pad_x, pad_y, args.size) for row in rows) + ("\n" if rows else ""),
                encoding="utf-8",
            )
            total += 1
            if index % 100 == 0 or index == len(images):
                print(f"{split}: {index}/{len(images)}")

    config = {
        "path": str(output),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": source_config["names"],
        "letterbox_size": args.size,
        "letterbox_fill": [114, 114, 114],
    }
    (output / "data.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"已生成 {total} 张等比例 letterbox 缓存图像：{output}")


if __name__ == "__main__":
    main()

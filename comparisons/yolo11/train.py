"""YOLO11n-cls 的 X-SDD 七分类训练入口。"""
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from comparisons.yolo_classification import main


if __name__ == "__main__":
    # 公平结构对比默认从 yolo11n-cls.yaml 随机初始化，不加载 ImageNet 权重。
    # 数据默认使用 prepare_xsdd.py 生成的固定 train/val/test 划分；这样从
    # PyCharm 直接运行本文件时不需要再手动填写参数。
    main(
        "11",
        default_pretrained=False,
        default_data=ROOT / "datasets" / "xsdd_yolo11_classification",
        default_run_prefix="xsdd",
    )

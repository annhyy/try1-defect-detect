"""YOLO26n-cls 的 X-SDD 七分类训练入口。"""
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from comparisons.yolo_classification import main


if __name__ == "__main__":
    # 与 YOLO11、DNM 和普通卷积对照一样，正式结构比较默认随机初始化。
    main(
        "26",
        default_pretrained=False,
        default_data=ROOT / "datasets" / "xsdd_yolo11_classification",
        default_run_prefix="xsdd",
    )

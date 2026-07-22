"""参数量更大的 YOLO11s APSPC 目标检测对照入口。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detection.yolo_train import main


if __name__ == "__main__":
    main(default_scale="s", default_out_name="yolo11s")

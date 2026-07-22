"""参数量近似匹配的普通卷积 APSPC 检测对照。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detection.dnm_train import main


if __name__ == "__main__":
    main(default_variant="conv", default_out=ROOT / "run2" / "controlled" / "conv_control", default_branch_features=8)

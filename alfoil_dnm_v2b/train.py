"""DNM-V2b 的 APSPC 目标检测入口。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detection.dnm_train import main


if __name__ == "__main__":
    main(default_variant="v2b", default_out=ROOT / "run2" / "controlled" / "dnm_v2b", default_branch_features=8)

"""DNM-V2b：X-SDD 分类、log 域几何平均的独立训练入口。"""
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alfoil_dnm.train import main


if __name__ == "__main__":
    main(
        default_variant="v2b",
        default_out=ROOT / "runs1" / "controlled" / "xsdd_dnm_v2b_cls",
        default_branch_features=8,
    )

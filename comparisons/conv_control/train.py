"""参数量与 DNM-V2 近似匹配的普通卷积消融入口。"""
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alfoil_dnm.train import main


if __name__ == "__main__":
    main(
        default_variant="conv",
        default_out=ROOT / "runs" / "controlled" / "conv_control",
        default_branch_features=8,
    )

"""Independent entry point for the DNM-V2b-F4 X-SDD ablation."""
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alfoil_dnm_next.train import main


if __name__ == "__main__":
    main(default_variant="v2b_f4")

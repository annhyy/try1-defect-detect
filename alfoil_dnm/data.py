from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset


def load_data_yaml(path: str | Path) -> Dict:
    path = Path(path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = Path(cfg.get("path", path.parent))
    if not root.is_absolute():
        root = (path.parent / root).resolve()
    cfg["root"] = root
    cfg["names"] = list(cfg["names"].values()) if isinstance(cfg["names"], dict) else cfg["names"]
    return cfg


class YoloDefectDataset(Dataset):
    """Read a standard YOLO detection split selected by ``data.yaml``.

    ``demo_alfoil/data.yaml`` selects the bundled synthetic smoke-test data;
    ``datasets/apspc_yolo/data.yaml`` selects the locally converted APSPC data.
    Both use identical image/label directory conventions, so no model code
    changes are required when switching from the demo to real data.
    """
    def __init__(self, cfg: Dict, split: str, image_size: int = 640) -> None:
        self.image_size = image_size
        image_dir = cfg["root"] / cfg[split]
        self.images = sorted(p for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp") for p in image_dir.rglob(ext))
        if not self.images:
            raise FileNotFoundError(f"No images found in {image_dir}")
        self.label_dir = cfg["root"] / "labels" / split

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        path = self.images[index]
        image = Image.open(path).convert("RGB").resize((self.image_size, self.image_size))
        image_tensor = torch.from_numpy(np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0)
        # Each image has a same-stem YOLO TXT label. Empty labels are valid
        # negative samples and remain represented as a [0, 5] tensor.
        label_path = self.label_dir / f"{path.stem}.txt"
        rows: List[List[float]] = []
        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                values = line.split()
                if len(values) == 5:
                    rows.append([float(value) for value in values])
        return image_tensor, torch.tensor(rows, dtype=torch.float32).reshape(-1, 5), str(path)


def collate(batch):
    images, targets, paths = zip(*batch)
    return torch.stack(images), list(targets), list(paths)

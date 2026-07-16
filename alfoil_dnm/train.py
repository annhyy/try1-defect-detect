from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Support both `python -m alfoil_dnm.train` and directly running this file in an IDE.
try:
    from .data import YoloDefectDataset, collate, load_data_yaml
    from .loss import detector_loss
    from .model import DendriticDetector
except ImportError:  # __package__ is empty when PyCharm/VS Code runs train.py directly
    from data import YoloDefectDataset, collate, load_data_yaml
    from loss import detector_loss
    from model import DendriticDetector


def run_epoch(model, loader, optimizer, num_classes, device):
    model.train(optimizer is not None)
    total = 0.0
    for images, targets, _ in loader:
        images = images.to(device)
        with torch.set_grad_enabled(optimizer is not None):
            prediction = model(images)
            loss, _ = detector_loss(prediction, targets, num_classes)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        total += loss.item() * images.shape[0]
    return total / len(loader.dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--branches", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="")
    parser.add_argument("--out", default="runs/alfoil_dnm")
    args = parser.parse_args()
    torch.manual_seed(42)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    # The YAML path is the single switch between the bundled demo and APSPC.
    cfg = load_data_yaml(args.data)
    train_set, val_set = YoloDefectDataset(cfg, "train", args.img_size), YoloDefectDataset(cfg, "val", args.img_size)
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=device.type == "cuda", collate_fn=collate)
    val_loader = DataLoader(val_set, args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda", collate_fn=collate)
    model = DendriticDetector(len(cfg["names"]), args.width, args.branches).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, len(cfg["names"]), device)
        val_loss = run_epoch(model, val_loader, None, len(cfg["names"]), device)
        scheduler.step()
        print(f"epoch {epoch:03d}/{args.epochs} train={train_loss:.4f} val={val_loss:.4f}")
        checkpoint = {"model": model.state_dict(), "names": cfg["names"], "width": args.width, "branches": args.branches, "img_size": args.img_size}
        torch.save(checkpoint, out / "last.pt")
        if val_loss < best:
            best = val_loss; torch.save(checkpoint, out / "best.pt")


if __name__ == "__main__":
    main()

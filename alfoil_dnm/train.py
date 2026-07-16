from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# 同时兼容 ``python -m alfoil_dnm.train`` 与 IDE 直接运行 train.py。
try:
    from .data import YoloDefectDataset, collate, load_data_yaml
    from .loss import detector_loss
    from .metrics import evaluate_map50
    from .model import DendriticDetector
except ImportError:  # IDE 直接运行时 __package__ 为空，改用当前目录导入。
    from data import YoloDefectDataset, collate, load_data_yaml
    from loss import detector_loss
    from metrics import evaluate_map50
    from model import DendriticDetector


def run_epoch(model, loader, optimizer, num_classes, device):
    """运行一个训练或验证 epoch；``optimizer=None`` 时只做验证。"""
    model.train(optimizer is not None)
    totals = {"loss": 0.0, "obj": 0.0, "box": 0.0, "cls": 0.0}
    for images, targets, _ in loader:
        images = images.to(device)
        with torch.set_grad_enabled(optimizer is not None):
            prediction = model(images)
            loss, details = detector_loss(prediction, targets, num_classes)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # 乘性树突分支可能放大局部梯度，裁剪可提升训练稳定性。
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        for key in totals:
            totals[key] += float(details[key]) * images.shape[0]
    return {key: value / len(loader.dataset) for key, value in totals.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--branches", type=int, default=4)
    parser.add_argument("--branch-features", type=int, default=4, help="每个树突分支参与乘性整合的局部特征数")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="")
    parser.add_argument("--out", default="runs/alfoil_dnm")
    parser.add_argument("--map-interval", type=int, default=5, help="每隔多少个 epoch 在验证集计算一次 mAP@0.5")
    args = parser.parse_args()
    torch.manual_seed(42)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    # ``--data`` 是数据源唯一入口；正式 APSPC 使用 datasets/apspc_yolo/data.yaml。
    cfg = load_data_yaml(args.data)
    train_set, val_set = YoloDefectDataset(cfg, "train", args.img_size), YoloDefectDataset(cfg, "val", args.img_size)
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=device.type == "cuda", collate_fn=collate)
    val_loader = DataLoader(val_set, args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda", collate_fn=collate)
    model = DendriticDetector(len(cfg["names"]), args.width, args.branches, args.branch_features).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, len(cfg["names"]), device)
        val_metrics = run_epoch(model, val_loader, None, len(cfg["names"]), device)
        scheduler.step()
        map50 = None
        if epoch % args.map_interval == 0 or epoch == args.epochs:
            map50 = evaluate_map50(model, val_loader, len(cfg["names"]), device)
        map_text = "--" if map50 is None else f"{map50:.4f}"
        print(
            f"epoch {epoch:03d}/{args.epochs} "
            f"train(total={train_metrics['loss']:.4f}, obj={train_metrics['obj']:.4f}, box={train_metrics['box']:.4f}, cls={train_metrics['cls']:.4f}) "
            f"val(total={val_metrics['loss']:.4f}, obj={val_metrics['obj']:.4f}, box={val_metrics['box']:.4f}, cls={val_metrics['cls']:.4f}) "
            f"mAP50={map_text}"
        )
        checkpoint = {"model": model.state_dict(), "names": cfg["names"], "width": args.width, "branches": args.branches, "branch_features": args.branch_features, "img_size": args.img_size, "map50": map50}
        torch.save(checkpoint, out / "last.pt")
        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            torch.save(checkpoint, out / "best.pt")


if __name__ == "__main__":
    main()

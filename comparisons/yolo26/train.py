"""YOLO26n 的 APSPC 目标检测基线。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from comparisons.control import controlled_yolo_options, standardize_yolo_metrics, write_protocol


def main() -> None:
    """运行 YOLO26n 的统一 APSPC 检测协议。"""
    parser = argparse.ArgumentParser(description="YOLO26n APSPC 目标检测")
    parser.add_argument("--data", default=str(ROOT / "datasets" / "apspc_yolo_letterbox640" / "data.yaml"))
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained", action="store_true")
    args = parser.parse_args()
    data_path = Path(args.data).resolve()
    data_config = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    cached_size = data_config.get("letterbox_size")
    if cached_size is not None and int(cached_size) != args.img_size:
        raise ValueError(
            f"数据已预先 letterbox 为 {cached_size}，不能再按 {args.img_size} 二次缩放。"
        )
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise SystemExit("缺少 ultralytics，请在当前 pytorch 环境安装。") from error
    source = "yolo26n.pt" if args.pretrained else "yolo26n.yaml"
    run_name = "yolo26n_pretrained" if args.pretrained else "yolo26n"
    model = YOLO(source)
    memory: dict[int, float] = {}

    def reset_peak(trainer) -> None:
        """每轮开始前清零峰值显存计数。"""
        if trainer.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(trainer.device)

    def record_peak(trainer) -> None:
        """记录本轮训练和验证过程的峰值显存。"""
        if trainer.device.type == "cuda":
            memory[trainer.epoch + 1] = torch.cuda.max_memory_allocated(trainer.device) / 1024 ** 2

    model.add_callback("on_train_epoch_start", reset_peak)
    model.add_callback("on_fit_epoch_end", record_peak)
    controlled = controlled_yolo_options(args.epochs, args.batch_size)
    controlled["workers"] = args.workers
    model.train(
        data=str(data_path), epochs=args.epochs, imgsz=args.img_size,
        batch=args.batch_size, device=args.device, seed=args.seed,
        pretrained=args.pretrained, project=str(ROOT / "run2" / "controlled"),
        name=run_name, exist_ok=True, **controlled,
    )
    save_dir = Path(model.trainer.save_dir)
    parameter_count = sum(parameter.numel() for parameter in model.model.parameters())
    best = YOLO(save_dir / "weights" / "best.pt")
    test = best.val(
        data=str(data_path), split="test", imgsz=args.img_size,
        batch=args.batch_size, device=args.device, workers=args.workers, amp=False,
        plots=True, verbose=False,
    )
    protocol = {
        "protocol": "apspc_detection_yolo_v1", "task": "object_detection",
        "model": "yolo26n", "model_source": source,
        "data": str(data_path), "epochs": args.epochs,
        "img_size": args.img_size, "batch_size": args.batch_size,
        "seed": args.seed, "pretrained": args.pretrained,
        "parameters": parameter_count, **controlled,
    }
    write_protocol(save_dir, protocol)
    standardize_yolo_metrics(save_dir / "results.csv", save_dir / "comparison_metrics.csv", memory)
    summary = {**protocol, "test": test.results_dict}
    (save_dir / "test_metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

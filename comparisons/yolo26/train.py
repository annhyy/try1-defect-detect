"""YOLO26n 与树突检测器的受控从零训练对照。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from comparisons.control import controlled_yolo_options, standardize_yolo_metrics, write_protocol


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO26n：受控树突检测对照实验")
    parser.add_argument("--data", default=str(ROOT / "datasets" / "apspc_yolo_letterbox640" / "data.yaml"))
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained", action="store_true", help="仅用于单独迁移学习实验；严格对照请勿设置")
    args = parser.parse_args()
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise SystemExit("缺少支持 YOLO26 的最新版 ultralytics。请执行：python -m pip install -U ultralytics") from error

    model_source = "yolo26n.pt" if args.pretrained else "yolo26n.yaml"
    run_name = "yolo26n_pretrained" if args.pretrained else "yolo26n"
    print(f"模型初始化：{model_source}；{'使用 COCO 预训练权重' if args.pretrained else '从零随机初始化'}")
    print("受控协议：增强=none；AdamW(lr=0.002, wd=0.0001)；余弦调度；AMP=False；名义批量=8")
    model = YOLO(model_source)
    # 记录每轮训练和验证期间的峰值已分配显存；口径与树突训练器一致。
    gpu_memory_mb_by_epoch: dict[int, float] = {}

    def reset_peak_memory(trainer) -> None:
        if trainer.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(trainer.device)

    def record_peak_memory(trainer) -> None:
        if trainer.device.type == "cuda":
            gpu_memory_mb_by_epoch[trainer.epoch + 1] = torch.cuda.max_memory_allocated(trainer.device) / 1024 ** 2

    model.add_callback("on_train_epoch_start", reset_peak_memory)
    model.add_callback("on_fit_epoch_end", record_peak_memory)
    controlled = controlled_yolo_options(args.epochs)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.img_size,
        batch=args.batch_size,
        device=args.device,
        seed=args.seed,
        pretrained=args.pretrained,
        project=str(ROOT / "runs" / "controlled"),
        name=run_name,
        exist_ok=True,
        **controlled,
    )
    # Ultralytics 的 train() 返回的是评价指标对象；实际输出目录由 trainer 保存。
    save_dir = Path(model.trainer.save_dir)
    parameter_count = sum(parameter.numel() for parameter in model.model.parameters())
    protocol = {
        "protocol": "controlled_scratch_v1" if not args.pretrained else "transfer_learning_v1",
        "model": "yolo26n",
        "model_source": model_source,
        "data": str(Path(args.data).resolve()),
        "epochs": args.epochs,
        "img_size": args.img_size,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "pretrained": args.pretrained,
        "augmentation": "none",
        "gradient_accumulation": 1,
        "parameters": parameter_count,
        **controlled,
    }
    write_protocol(save_dir, protocol)
    test_metrics = YOLO(save_dir / "weights" / "best.pt").val(
        data=args.data, split="test", imgsz=args.img_size, batch=args.batch_size,
        device=args.device, workers=2, amp=False,
    )
    summary = {**protocol, "test": test_metrics.results_dict}
    (save_dir / "test_metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    standardize_yolo_metrics(save_dir / "results.csv", save_dir / "comparison_metrics.csv", gpu_memory_mb_by_epoch)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"统一逐轮指标：{save_dir / 'comparison_metrics.csv'}")


if __name__ == "__main__":
    main()

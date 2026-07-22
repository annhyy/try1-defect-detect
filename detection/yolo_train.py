"""在 APSPC 目标检测数据上训练 Ultralytics YOLO。

YOLO11n 与 YOLO11s 共用该入口。两组只改变模型规模，数据、输入尺寸、
优化器和增强设置保持一致。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from comparisons.control import controlled_yolo_options, standardize_yolo_metrics, write_protocol


def _parameter_count(model) -> int:
    """统计模型总参数量。"""
    return sum(parameter.numel() for parameter in model.parameters())


def main(default_scale: str = "n", default_out_name: str | None = None) -> None:
    """运行 YOLO11n 或 YOLO11s 的统一 APSPC 检测协议。"""
    if default_scale not in {"n", "s"}:
        raise ValueError("YOLO11 规模只能是 n 或 s")
    parser = argparse.ArgumentParser(description=f"YOLO11{default_scale} APSPC 目标检测")
    parser.add_argument("--data", default=str(ROOT / "datasets" / "apspc_yolo_letterbox640" / "data.yaml"))
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained", action="store_true", help="单独运行 COCO 迁移学习；严格结构对照默认从零训练")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()
    data_path = Path(args.data).resolve()
    data_config = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    cached_size = data_config.get("letterbox_size")
    if cached_size is not None and int(cached_size) != args.img_size:
        raise ValueError(
            f"数据已预先 letterbox 为 {cached_size}，不能再按 {args.img_size} 二次缩放。"
            "若要测试更大尺寸，请改用 datasets/apspc_yolo/data.yaml。"
        )
    if args.img_size != 640:
        print(f"注意：当前受控协议建议 640×640，实际使用 {args.img_size}×{args.img_size}")

    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise SystemExit("缺少 ultralytics，请在当前 pytorch 环境安装。") from error

    source = f"yolo11{default_scale}.pt" if args.pretrained else f"yolo11{default_scale}.yaml"
    base_name = default_out_name or f"yolo11{default_scale}"
    run_name = args.run_name or (f"{base_name}_pretrained" if args.pretrained else base_name)
    print(f"任务：APSPC 目标检测；模型：{source}；输入：{args.img_size}×{args.img_size}")
    print(f"初始化：{'COCO 预训练' if args.pretrained else '随机初始化'}；结果：run2/controlled/{run_name}")

    model = YOLO(source)
    gpu_memory_by_epoch: dict[int, float] = {}
    def on_epoch_start(trainer) -> None:
        """每轮开始前清零峰值显存计数。"""
        if trainer.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(trainer.device)

    def on_epoch_end(trainer) -> None:
        """记录本轮训练和验证过程的峰值显存。"""
        epoch = trainer.epoch + 1
        if epoch > args.epochs:
            return
        if trainer.device.type == "cuda":
            gpu_memory_by_epoch[epoch] = torch.cuda.max_memory_allocated(trainer.device) / 1024 ** 2

    model.add_callback("on_train_epoch_start", on_epoch_start)
    model.add_callback("on_fit_epoch_end", on_epoch_end)
    controlled = controlled_yolo_options(args.epochs, args.batch_size)
    controlled["workers"] = args.workers
    import time

    training_start = time.perf_counter()
    model.train(
        data=str(data_path),
        epochs=args.epochs, imgsz=args.img_size, batch=args.batch_size,
        device=args.device, seed=args.seed,
        pretrained=args.pretrained, project=str(ROOT / "run2" / "controlled"),
        name=run_name, exist_ok=True, **controlled,
    )
    training_seconds = time.perf_counter() - training_start
    save_dir = Path(model.trainer.save_dir)
    best_path = save_dir / "weights" / "best.pt"
    if not best_path.exists():
        raise FileNotFoundError(f"Ultralytics 未生成最优权重：{best_path}")
    best_wrapper = YOLO(best_path)
    raw_model = best_wrapper.model
    parameter_count = _parameter_count(raw_model)
    test_results = best_wrapper.val(
        data=str(data_path), split="test", imgsz=args.img_size,
        batch=args.batch_size, device=args.device, workers=args.workers, amp=False,
        plots=True, verbose=False,
    )
    protocol = {
        "protocol": "apspc_detection_yolo_v1",
        "task": "object_detection",
        "model": f"yolo11{default_scale}",
        "model_source": source,
        "data": str(data_path),
        "epochs": args.epochs, "img_size": args.img_size, "batch_size": args.batch_size,
        "seed": args.seed, "pretrained": args.pretrained,
        "parameters": parameter_count, "training_seconds": training_seconds,
        **controlled,
    }
    write_protocol(save_dir, protocol)
    standardize_yolo_metrics(
        save_dir / "results.csv", save_dir / "comparison_metrics.csv", gpu_memory_by_epoch
    )
    summary = {**protocol, "test": test_results.results_dict}
    (save_dir / "test_metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"统一指标：{save_dir / 'comparison_metrics.csv'}")


if __name__ == "__main__":
    main()

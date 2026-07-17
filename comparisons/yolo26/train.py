"""使用 Ultralytics YOLO26n 在 APSPC YOLO 数据上进行对比训练。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO26 与树突检测器的公平对比实验")
    # 与树突模型使用同一份等比例 letterbox 缓存数据，避免预处理差异影响对照结论。
    parser.add_argument("--data", default=str(ROOT / "datasets" / "apspc_yolo_letterbox640" / "data.yaml"))
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained", action="store_true", help="使用 COCO 预训练 yolo26n.pt；公平从零训练时不要设置")
    args = parser.parse_args()
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise SystemExit("缺少支持 YOLO26 的最新版 ultralytics。请执行：python -m pip install -U ultralytics") from error
    model_source = "yolo26n.pt" if args.pretrained else "yolo26n.yaml"
    print(f"模型初始化：{model_source}；{'使用 COCO 预训练权重' if args.pretrained else '从零随机初始化'}")
    model = YOLO(model_source)
    result = model.train(data=args.data, epochs=args.epochs, imgsz=args.img_size, batch=args.batch_size, device=args.device, seed=args.seed, pretrained=args.pretrained, project=str(ROOT / "comparisons" / "yolo26" / "runs"), name="apspc", exist_ok=True)
    metrics = YOLO(Path(result.save_dir) / "weights" / "best.pt").val(data=args.data, split="test", imgsz=args.img_size, batch=args.batch_size, device=args.device)
    summary = {"model": "yolo26n", "pretrained": args.pretrained, "data": args.data, "results": metrics.results_dict}
    (Path(result.save_dir) / "test_metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

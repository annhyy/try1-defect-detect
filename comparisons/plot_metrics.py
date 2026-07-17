"""绘制树突消融模型、普通卷积和 YOLO 的可比检测指标。"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        return [{key.strip(): value.strip() for key, value in row.items() if key} for row in csv.DictReader(file)]


def number(row: dict[str, str], name: str) -> float | None:
    value = row.get(name, "")
    return float(value) if value else None


def series(path: Path, name: str) -> dict:
    rows = read_csv(path)
    return {"name": name, "epoch": [number(row, "epoch") for row in rows], **{key: [number(row, key) for row in rows] for key in ("precision", "recall", "map50", "map50_95")}}


def plot_line(axis, items: list[dict], key: str, title: str) -> None:
    for item in items:
        points = [(epoch, value) for epoch, value in zip(item["epoch"], item[key]) if epoch is not None and value is not None]
        if points:
            epochs, values = zip(*points)
            axis.plot(epochs, values, marker="o", markersize=2, linewidth=1.5, label=item["name"])
    axis.set_title(title)
    axis.set_xlabel("Epoch")
    axis.set_ylim(0, 1)
    axis.grid(alpha=0.25)
    if axis.lines:
        axis.legend()


def main() -> None:
    parser = argparse.ArgumentParser(description="绘制 DNM 消融、普通卷积和 YOLO 的受控检测指标对照图")
    parser.add_argument("--dnm", default=str(ROOT / "runs" / "controlled" / "dnm" / "comparison_metrics.csv"))
    parser.add_argument("--dnm-v2a", default=str(ROOT / "runs" / "controlled" / "dnm_v2a" / "comparison_metrics.csv"))
    parser.add_argument("--dnm-v2b", default=str(ROOT / "runs" / "controlled" / "dnm_v2b" / "comparison_metrics.csv"))
    parser.add_argument("--conv-control", default=str(ROOT / "runs" / "controlled" / "conv_control" / "comparison_metrics.csv"))
    parser.add_argument("--yolo11", default=str(ROOT / "runs" / "controlled" / "yolo11n" / "comparison_metrics.csv"))
    parser.add_argument("--yolo26", default=str(ROOT / "runs" / "controlled" / "yolo26n" / "comparison_metrics.csv"))
    parser.add_argument("--out", default=str(ROOT / "runs" / "controlled" / "metrics_comparison.png"))
    args = parser.parse_args()
    sources = (
        (Path(args.dnm), "DNM-V1"),
        (Path(args.dnm_v2a), "DNM-V2a product"),
        (Path(args.dnm_v2b), "DNM-V2b geometric mean"),
        (Path(args.conv_control), "Conv control"),
        (Path(args.yolo11), "YOLO11n"),
        (Path(args.yolo26), "YOLO26n"),
    )
    items = [series(path, name) for path, name in sources if path.exists()]
    for path, _ in sources:
        if not path.exists():
            print(f"跳过尚不存在的日志：{path}")
    if not items:
        raise FileNotFoundError("未找到 comparison_metrics.csv；请先完成至少一个受控训练。")

    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    plot_line(axes[0, 0], items, "precision", "Validation Precision")
    plot_line(axes[0, 1], items, "recall", "Validation Recall")
    plot_line(axes[1, 0], items, "map50", "Validation mAP@0.5")
    plot_line(axes[1, 1], items, "map50_95", "Validation mAP@0.5:0.95")
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    print(f"已生成可比指标图：{output.resolve()}")


if __name__ == "__main__":
    main()

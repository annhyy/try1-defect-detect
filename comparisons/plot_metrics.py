"""绘制可横向比较的 APSPC 目标检测指标。"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
METRICS = (
    ("precision", "Validation Precision"),
    ("recall", "Validation Recall"),
    ("map50", "Validation mAP@0.5"),
    ("map50_95", "Validation mAP@0.5:0.95"),
)
SOURCES = (
    ("dnm_v1", "DNM-V1"),
    ("dnm_v2a", "DNM-V2a"),
    ("dnm_v2b", "DNM-V2b"),
    ("conv_control", "Conv-Control"),
    ("yolo11n", "YOLO11n"),
    ("yolo11s", "YOLO11s"),
    ("yolo26n", "YOLO26n"),
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        return [
            {key.strip(): value.strip() for key, value in row.items() if key}
            for row in csv.DictReader(file)
        ]


def number(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "")
    return float(value) if value else None


def curve(path: Path, name: str) -> dict:
    rows = read_csv(path)
    return {
        "name": name,
        "epoch": [number(row, "epoch") for row in rows],
        **{key: [number(row, key) for row in rows] for key, _ in METRICS},
    }


def plot_line(axis, series: list[dict], key: str, title: str) -> None:
    for item in series:
        points = [
            (epoch, value)
            for epoch, value in zip(item["epoch"], item[key])
            if epoch is not None and value is not None
        ]
        if points:
            epochs, values = zip(*points)
            axis.plot(epochs, values, linewidth=1.6, label=item["name"])
    axis.set_title(title)
    axis.set_xlabel("Epoch")
    axis.set_ylim(0, 1.01)
    axis.grid(alpha=0.25)
    if axis.lines:
        axis.legend(fontsize=8)


def test_metric(summary: dict, key: str) -> float:
    test = summary["test"]
    aliases = {
        "precision": ("precision", "metrics/precision(B)", "metrics/precision"),
        "recall": ("recall", "metrics/recall(B)", "metrics/recall"),
        "map50": ("map50", "metrics/mAP50(B)", "metrics/mAP50"),
        "map50_95": ("map50_95", "metrics/mAP50-95(B)", "metrics/mAP50-95"),
    }
    for alias in aliases[key]:
        if alias in test:
            return float(test[alias])
    raise KeyError(f"test_metrics.json 缺少 {key}，现有字段：{sorted(test)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="绘制 APSPC 目标检测统一指标")
    parser.add_argument("--root", default=str(ROOT / "run2" / "controlled"))
    args = parser.parse_args()
    run_root = Path(args.root).resolve()
    available = [
        (run_root / folder, name)
        for folder, name in SOURCES
        if (run_root / folder / "comparison_metrics.csv").exists()
    ]
    if not available:
        raise FileNotFoundError(f"{run_root} 中没有 comparison_metrics.csv")

    import matplotlib.pyplot as plt

    run_root.mkdir(parents=True, exist_ok=True)
    series = [curve(folder / "comparison_metrics.csv", name) for folder, name in available]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for axis, (key, title) in zip(axes.flat, METRICS):
        plot_line(axis, series, key, title)
    curve_path = run_root / "metrics_comparison.png"
    figure.savefig(curve_path, dpi=180)
    plt.close(figure)

    summaries = []
    for folder, name in available:
        path = folder / "test_metrics.json"
        if path.exists():
            summaries.append((name, json.loads(path.read_text(encoding="utf-8"))))
    final_path = None
    if summaries:
        names = [name for name, _ in summaries]
        final, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        for axis, (key, title) in zip(axes.flat, METRICS):
            values = [test_metric(summary, key) for _, summary in summaries]
            axis.bar(names, values)
            axis.set_title(title.replace("Validation", "Test"))
            axis.set_ylim(0, 1.01)
            axis.tick_params(axis="x", rotation=25)
            axis.grid(axis="y", alpha=0.2)
        final_path = run_root / "final_detection_comparison.png"
        final.savefig(final_path, dpi=180)
        plt.close(final)

    print(f"检测训练曲线：{curve_path}")
    if final_path:
        print(f"测试集检测指标：{final_path}")


if __name__ == "__main__":
    main()

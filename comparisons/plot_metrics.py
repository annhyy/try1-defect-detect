"""绘制 X-SDD 分类曲线与最终准确率、参数量、推理速度对比。"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        return [{key.strip(): value.strip() for key, value in row.items() if key} for row in csv.DictReader(file)]


def number(row: dict[str, str], name: str) -> float | None:
    value = row.get(name, "")
    return float(value) if value else None


def curve_series(path: Path, name: str) -> dict:
    rows = read_csv(path)
    keys = ("train_loss", "val_loss", "val_accuracy", "val_macro_f1")
    return {
        "name": name,
        "epoch": [number(row, "epoch") for row in rows],
        **{key: [number(row, key) for row in rows] for key in keys},
    }


def plot_line(axis, items: list[dict], key: str, title: str, unit_interval: bool = False) -> None:
    for item in items:
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
    if unit_interval:
        axis.set_ylim(0, 1.01)
    axis.grid(alpha=0.25)
    if axis.lines:
        axis.legend(fontsize=8)


def main() -> None:
    parser = argparse.ArgumentParser(description="绘制 DNM、普通分类头与 YOLO 的统一分类指标")
    parser.add_argument("--root", default=str(ROOT / "runs1" / "controlled"))
    args = parser.parse_args()
    run_root = Path(args.root)
    sources = (
        (run_root / "xsdd_dnm_v1_cls", "DNM-V1"),
        (run_root / "xsdd_dnm_v2a_cls", "DNM-V2a"),
        (run_root / "xsdd_dnm_v2b_cls", "DNM-V2b"),
        (run_root / "xsdd_dnm_v2a_f4_cls", "DNM-V2a-F4"),
        (run_root / "xsdd_dnm_v2b_f4_cls", "DNM-V2b-F4"),
        (run_root / "xsdd_dnm_v1_tuned_cls", "DNM-V1-Tuned"),
        (run_root / "xsdd_conv_control_cls", "Conv control"),
        (run_root / "xsdd_conv_control_weighted_cls", "Conv control weighted"),
        (run_root / "xsdd_yolo11n_cls_scratch", "YOLO11n-cls"),
        (run_root / "xsdd_yolo26n_cls_scratch", "YOLO26n-cls"),
    )
    items = [
        curve_series(folder / "comparison_metrics.csv", name)
        for folder, name in sources
        if (folder / "comparison_metrics.csv").exists()
    ]
    if not items:
        raise FileNotFoundError("runs1 中还没有 comparison_metrics.csv")

    import matplotlib.pyplot as plt

    run_root.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    plot_line(axes[0, 0], items, "val_accuracy", "Validation Accuracy", True)
    plot_line(axes[0, 1], items, "val_macro_f1", "Validation Macro-F1", True)
    plot_line(axes[1, 0], items, "train_loss", "Training Loss")
    plot_line(axes[1, 1], items, "val_loss", "Validation Loss")
    curve_output = run_root / "metrics_comparison.png"
    figure.savefig(curve_output, dpi=180)
    plt.close(figure)

    summaries = []
    for folder, name in sources:
        path = folder / "test_metrics.json"
        if path.exists():
            summaries.append((name, json.loads(path.read_text(encoding="utf-8"))))
    if summaries:
        names = [name for name, _ in summaries]
        accuracy = [item["test"]["accuracy"] for _, item in summaries]
        macro_f1 = [item["test"]["macro_f1"] for _, item in summaries]
        parameters = [item["parameters"] / 1e6 for _, item in summaries]
        latency = []
        for _, item in summaries:
            speed = item.get("speed_batch1_forward", {})
            selected = speed.get("gpu", speed.get("cpu", {}))
            latency.append(selected.get("median_ms", 0.0))

        final, bars = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        for axis, values, title in (
            (bars[0, 0], accuracy, "Test Accuracy"),
            (bars[0, 1], macro_f1, "Test Macro-F1"),
            (bars[1, 0], parameters, "Parameters (M)"),
            (bars[1, 1], latency, "Batch-1 Forward Latency (ms)"),
        ):
            axis.bar(names, values)
            axis.set_title(title)
            axis.tick_params(axis="x", rotation=25)
            axis.grid(axis="y", alpha=0.2)
        final.savefig(run_root / "final_comparison.png", dpi=180)
        plt.close(final)

    print(f"已生成分类训练曲线：{curve_output.resolve()}")


if __name__ == "__main__":
    main()

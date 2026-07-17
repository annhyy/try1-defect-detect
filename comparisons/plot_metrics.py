"""读取树突模型与 Ultralytics YOLO 的日志，生成可直接用于对照实验的指标图。"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> list[dict[str, str]]:
    """读取 CSV，并移除 Ultralytics 列名中偶尔出现的空格。"""
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        return [{key.strip(): value.strip() for key, value in row.items() if key} for row in csv.DictReader(file)]


def number(row: dict[str, str], *names: str) -> float | None:
    """从一行日志中读取第一个存在且非空的数值列。"""
    for name in names:
        value = row.get(name, "")
        if value:
            return float(value)
    return None


def normalized(values: list[float | None]) -> list[float | None]:
    """用首个有效值归一化 loss；不同模型的 loss 公式不同，不能直接比较绝对值。"""
    first = next((value for value in values if value is not None and value != 0), None)
    return [value / first if value is not None and first is not None else None for value in values]


def dnm_series(path: Path) -> dict:
    rows = read_csv(path)
    return {
        "name": "Dendritic detector",
        "epoch": [number(row, "epoch") for row in rows],
        "loss": [number(row, "train_total") for row in rows],
        "map50": [number(row, "map50") for row in rows],
        # 当前树突评估器只实现 mAP@0.5，因此此处明确保留为空而不是伪造 mAP50-95。
        "map5095": [None] * len(rows),
    }


def yolo_series(path: Path, name: str) -> dict:
    rows = read_csv(path)
    loss = []
    for row in rows:
        # Ultralytics 各版本都以这三个检测损失列记录；缺失的列按 0 处理。
        components = [number(row, "train/box_loss") or 0.0, number(row, "train/cls_loss") or 0.0, number(row, "train/dfl_loss") or 0.0]
        loss.append(sum(components))
    return {
        "name": name,
        "epoch": [number(row, "epoch") for row in rows],
        "loss": loss,
        "map50": [number(row, "metrics/mAP50(B)", "metrics/mAP50") for row in rows],
        "map5095": [number(row, "metrics/mAP50-95(B)", "metrics/mAP50-95") for row in rows],
    }


def plot_line(axis, series: list[dict], key: str, ylabel: str) -> None:
    for item in series:
        points = [(epoch, value) for epoch, value in zip(item["epoch"], item[key]) if epoch is not None and value is not None]
        if points:
            epochs, values = zip(*points)
            axis.plot(epochs, values, marker="o", markersize=2, linewidth=1.5, label=item["name"])
    axis.set_xlabel("Epoch")
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.25)
    if axis.lines:
        axis.legend()


def main() -> None:
    parser = argparse.ArgumentParser(description="绘制 DNM、YOLO11、YOLO26 的训练指标对照图")
    parser.add_argument("--dnm", default=str(ROOT / "runs" / "apspc_dnm_letterbox640" / "metrics.csv"))
    parser.add_argument("--yolo11", default=str(ROOT / "comparisons" / "yolo11" / "runs" / "apspc" / "results.csv"))
    parser.add_argument("--yolo26", default=str(ROOT / "comparisons" / "yolo26" / "runs" / "apspc" / "results.csv"))
    parser.add_argument("--out", default=str(ROOT / "comparisons" / "figures" / "metrics_comparison.png"))
    args = parser.parse_args()

    sources = ((Path(args.dnm), dnm_series, ()), (Path(args.yolo11), yolo_series, ("YOLO11n",)), (Path(args.yolo26), yolo_series, ("YOLO26n",)))
    series = []
    for path, reader, extra in sources:
        if path.exists():
            series.append(reader(path, *extra))
        else:
            print(f"跳过尚不存在的日志：{path}")
    if not series:
        raise FileNotFoundError("未找到任何训练日志；请先完成至少一个模型的训练。")

    # 延迟导入，使脚本在仅查看 --help 时不依赖 matplotlib。
    import matplotlib.pyplot as plt

    for item in series:
        item["loss"] = normalized(item["loss"])
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
    plot_line(axes[0], series, "loss", "Normalized training loss")
    plot_line(axes[1], series, "map50", "Validation mAP@0.5")
    plot_line(axes[2], series, "map5095", "Validation mAP@0.5:0.95")
    axes[0].set_title("Convergence trend")
    axes[1].set_title("Comparable detection metric")
    axes[2].set_title("YOLO metric (DNM not implemented)")
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    print(f"已生成对照图：{output.resolve()}")


if __name__ == "__main__":
    main()

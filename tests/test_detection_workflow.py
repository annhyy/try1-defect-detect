"""目标检测入口的日志与几何预处理回归测试。"""
from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from alfoil_dnm.infer import letterbox_image, restore_box
from comparisons.control import standardize_yolo_metrics


class DetectionWorkflowTests(unittest.TestCase):
    def test_letterbox_box_round_trip(self) -> None:
        image = Image.new("RGB", (2560, 1920))
        cached, scale, pad_x, pad_y = letterbox_image(image, 640)
        self.assertEqual(cached.size, (640, 640))
        self.assertAlmostEqual(scale, 0.25)
        self.assertEqual((pad_x, pad_y), (0, 80))
        restored = restore_box([25, 130, 225, 330], scale, pad_x, pad_y, 2560, 1920)
        self.assertEqual(restored, [100.0, 200.0, 900.0, 1000.0])

    def test_yolo_metric_columns_and_epoch_numbers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "results.csv"
            source.write_text(
                "epoch,time,metrics/precision(B),metrics/recall(B),"
                "metrics/mAP50(B),metrics/mAP50-95(B),lr/pg0\n"
                "1,10,0.1,0.2,0.3,0.4,0.002\n"
                "2,22,0.2,0.3,0.4,0.5,0.001\n",
                encoding="utf-8",
            )
            output = root / "comparison_metrics.csv"
            standardize_yolo_metrics(source, output, {1: 100.0, 2: 120.0})
            with output.open(encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual([row["epoch"] for row in rows], ["1", "2"])
            self.assertEqual(rows[1]["map50_95"], "0.5")
            self.assertEqual(rows[1]["epoch_seconds"], "12.0")


if __name__ == "__main__":
    unittest.main()

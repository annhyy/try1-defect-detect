# YOLO11 对比实验

YOLO11 是 Ultralytics 的主流稳定版本，支持检测训练、验证和导出。默认使用最小的 `yolo11n`，便于与轻量树突模型比较。

```powershell
# 受控的从零训练：无预训练、无增强、AdamW、同一数据/批量/训练轮数
python .\comparisons\yolo11\train.py --epochs 120 --img-size 640 --batch-size 8

# 使用 COCO 预训练权重的迁移学习（需要在报告中单独标注）
python .\comparisons\yolo11\train.py --pretrained --epochs 120 --img-size 640 --batch-size 8
```

默认加载 `datasets/apspc_yolo_letterbox640/data.yaml`，与树突模型使用相同的等比例缓存图像、标签和划分。受控结果写入 `runs/controlled/yolo11n/`：原生 `results.csv` 保留 YOLO 私有 loss，`comparison_metrics.csv` 只保存可横向比较的 Precision、Recall、mAP50、mAP50-95、时间和学习率，`test_metrics.json` 保存测试集最终指标。完成三组训练后，在仓库根目录运行 `python .\comparisons\plot_metrics.py` 生成统一指标图。

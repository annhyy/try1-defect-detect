# YOLO26 对比实验

YOLO26 是 Ultralytics 的较新一代模型，支持端到端、默认免 NMS 的检测推理。为控制算力，默认使用最小的 `yolo26n`。

```powershell
# 受控的从零训练：无预训练、无增强、AdamW、同一数据/批量/训练轮数
python .\comparisons\yolo26\train.py --epochs 120 --img-size 640 --batch-size 8

# 使用 COCO 预训练权重的迁移学习（需要在报告中单独标注）
python .\comparisons\yolo26\train.py --pretrained --epochs 120 --img-size 640 --batch-size 8
```

默认加载 `datasets/apspc_yolo_letterbox640/data.yaml`，与树突模型使用相同的等比例缓存图像、标签和划分。受控结果写入 `runs/controlled/yolo26n/`：原生 `results.csv` 保留 YOLO 私有 loss，`comparison_metrics.csv` 只保存可横向比较的 Precision、Recall、mAP50、mAP50-95、时间和学习率，`test_metrics.json` 保存测试集最终指标。完成三组训练后，在仓库根目录运行 `python .\comparisons\plot_metrics.py` 生成统一指标图。

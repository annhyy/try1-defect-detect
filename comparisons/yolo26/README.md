# YOLO26 对比实验

YOLO26 是 Ultralytics 的较新一代模型，支持端到端、默认免 NMS 的检测推理。为控制算力，默认使用最小的 `yolo26n`。

```powershell
# 公平的从零训练
python .\comparisons\yolo26\train.py --epochs 120 --img-size 640 --batch-size 8

# 使用 COCO 预训练权重的迁移学习（需要在报告中单独标注）
python .\comparisons\yolo26\train.py --pretrained --epochs 120 --img-size 640 --batch-size 8
```

结果写入 `comparisons/yolo26/runs/apspc/`：Ultralytics 的 `results.csv` 保存逐 epoch 损失和检测指标，`test_metrics.json` 保存测试集最终指标。

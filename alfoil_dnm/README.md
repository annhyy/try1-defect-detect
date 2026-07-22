# DNM-V1 APSPC 目标检测器

`train.py` 现在读取 APSPC 的 YOLO 框标注，训练现有单尺度 DNM 检测器。模型在
stride-8 特征图上输出目标置信度、归一化边框和 10 类 logits。本轮只是把任务恢复为
目标检测，没有加入“树突替换卷积”等新结构。

```powershell
D:\Anaconda_envs\envs\pytorch\python.exe .\alfoil_dnm\train.py
```

默认数据为 `datasets/apspc_yolo_letterbox640/data.yaml`，输入 640x640，训练
120 epoch、batch 8，结果写入 `run2/controlled/dnm_v1/`。最优检查点按验证集
mAP50-95 选择；`metrics.csv` 保存 objectness/box/class loss 和检测指标，
`test_metrics.json` 保存测试集 P/R/mAP 与逐类结果。

推理示例：

```powershell
D:\Anaconda_envs\envs\pytorch\python.exe .\alfoil_dnm\infer.py --weights .\run2\controlled\dnm_v1\best.pt --source image.jpg --data .\datasets\apspc_yolo_letterbox640\data.yaml --out prediction.jpg
```

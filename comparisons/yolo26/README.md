# YOLO26n：X-SDD 七分类基线

这里使用 Ultralytics `yolo26n-cls` 图像分类模型，不是目标检测模型。默认读取
`datasets/xsdd_yolo11_classification/`，从 `yolo26n-cls.yaml` 随机初始化，与
YOLO11、DNM 和普通卷积对照保持一致。

```powershell
D:\Anaconda_envs\envs\pytorch\python.exe .\comparisons\yolo26\train.py
```

默认结果：`runs1/controlled/xsdd_yolo26n_cls_scratch/`。只有显式添加
`--pretrained` 才加载 ImageNet 权重；预训练结果必须作为迁移学习参考单独报告。

逐轮和最终输出格式与 YOLO11 相同，包括 loss、Accuracy、Macro-P/R/F1、参数量、
CPU/GPU batch=1 推理速度、混淆矩阵和逐图预测。

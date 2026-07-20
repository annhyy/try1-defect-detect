# YOLO11n：X-SDD 七分类基线

这里运行的是 Ultralytics `YOLO11n-cls` **图像分类**模型，不是画缺陷框的检测模型。
每张图片只输出 7 个钢材表面缺陷类别中的一个。

## 1. 准备数据

原始数据应放在 `datasets/X-SDD/datas/<类别名>/`。第一次运行训练前执行：

```powershell
D:\Anaconda_envs\envs\pytorch\python.exe .\comparisons\yolo11\prepare_xsdd.py
```

脚本不会修改原始数据。它会计算 SHA-256、排除完全重复图片，并按类别以固定随机种子
划分 70% train、15% val、15% test，输出到
`datasets/xsdd_yolo11_classification/`。详细划分记录在：

- `dataset_info.json`：类别、数量、重复图和划分统计；
- `split_manifest.csv`：每张图片所属集合、类别和 SHA-256。

## 2. 训练

在 PyCharm 中直接运行 `comparisons/yolo11/train.py` 即可；默认配置为：

- `yolo11n-cls.yaml` 随机初始化，不下载或加载 ImageNet 预训练权重；
- 输入尺寸 224×224，100 epoch，batch size 64，GPU 0；
- 结果目录 `runs1/controlled/xsdd_yolo11n_cls_scratch/`。

等价命令为：

```powershell
D:\Anaconda_envs\envs\pytorch\python.exe .\comparisons\yolo11\train.py
```

如需迁移学习参考实验，显式加 `--pretrained`。正式公平结构对比仍使用默认的随机初始化。

训练过程中，Ultralytics 的原始 loss/Top-1 写入 `results.csv`；本项目每轮补算 Accuracy、
Macro-Precision、Macro-Recall 和 Macro-F1，写入 `comparison_metrics.csv`。训练结束后，
固定 test 集结果写入 `test_metrics.json`、`confusion_matrix.csv` 和
`test_predictions.csv`。

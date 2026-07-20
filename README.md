# X-SDD 表面缺陷分类：树突神经元与 YOLO 对比

当前主实验是 **X-SDD 七类钢材表面缺陷整图分类**，不是目标检测。模型接收整张
图片，只输出一个类别，不读取文件名、XML 或 YOLO 检测框。

七个类别为：`finishing roll printing`、`iron sheet ash`、
`oxide scale of plate system`、`oxide scale of temperature system`、
`red iron`、`slag inclusion`、`surface scratch`。

## 1. 数据准备

原始数据放在 `datasets/X-SDD/datas/<类别>/`。首次准备执行：

```powershell
D:\Anaconda_envs\envs\pytorch\python.exe .\comparisons\yolo11\prepare_xsdd.py
```

脚本保留原图，计算 SHA-256 并排除 9 张完全重复图，再以 seed 42 按类别划分
70%/15%/15%。最终得到 1351 张：train 946、val 201、test 204，输出目录为
`datasets/xsdd_yolo11_classification/`。所有模型读取这一份固定划分。

## 2. 正式训练

以下入口默认都是随机初始化、224×224、100 epoch、batch 64、GPU 0：

```powershell
# YOLO11n-cls
D:\Anaconda_envs\envs\pytorch\python.exe .\comparisons\yolo11\train.py

# YOLO26n-cls
D:\Anaconda_envs\envs\pytorch\python.exe .\comparisons\yolo26\train.py

# 旧版直接连乘 DNM
D:\Anaconda_envs\envs\pytorch\python.exe .\alfoil_dnm\train.py

# 论文公式 + log 域精确乘积
D:\Anaconda_envs\envs\pytorch\python.exe .\alfoil_dnm_v2a\train.py

# 论文公式 + log 域几何平均
D:\Anaconda_envs\envs\pytorch\python.exe .\alfoil_dnm_v2b\train.py

# 共享骨干、参数量近似匹配的普通神经网络分类头
D:\Anaconda_envs\envs\pytorch\python.exe .\comparisons\conv_control\train.py
```

只有显式加 `--pretrained` 时 YOLO 才加载 ImageNet 权重；该结果必须作为迁移学习
参考单独报告，不能与随机初始化的树突模型混为结构对比。

## 3. 结果目录与指标

所有正式结果写入 `runs1/controlled/xsdd_*`。每个实验保存：

- `comparison_metrics.csv`：逐 epoch 的 loss、验证 Accuracy、Macro-P/R/F1、时间、
  显存和学习率；
- `test_metrics.json`：固定测试集最终指标、参数量、权重大小和 CPU/GPU 推理延迟；
- `confusion_matrix.csv`：七类混淆矩阵；
- `test_predictions.csv`：逐图真值与预测；
- `best.pt` 或 `weights/best.pt`：验证 Accuracy 最佳权重。

训练全部结束后绘图：

```powershell
D:\Anaconda_envs\envs\pytorch\python.exe .\comparisons\plot_metrics.py
```

图保存到 `runs1/controlled/metrics_comparison.png` 和 `final_comparison.png`。

详细说明见 [X-SDD 数据说明](documents/xsdd_dataset.md)、
[统一对比协议](documents/comparison_protocol.md) 和
[树突消融结构](documents/dnm_ablation_models.md)。NEU 和 APSPC 的旧说明仅作为历史记录，
不属于当前 X-SDD 主实验。

## 项目结构

```text
classification/       分类共享加载器、模型、统一指标和旧 NEU 整理脚本
alfoil_dnm*/           DNM-V1、V2a、V2b 训练入口
comparisons/           普通网络、YOLO11/26、X-SDD 准备和绘图入口
documents/             数据与实验说明
datasets/              本地数据（Git 忽略）
runs1/                 当前分类结果（Git 忽略）
runs/                  旧检测结果（Git 忽略）
```

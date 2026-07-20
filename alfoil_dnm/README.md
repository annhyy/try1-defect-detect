# DNM-V1：X-SDD 七分类入口

`train.py` 当前训练 X-SDD 七类整图分类。共享轻量卷积骨干先把 224×224 图像变成
空间特征，经过全局平均池化后送入经典四层 DNM 分类头。V1 保留基础代码中的负号
突触形式和直接乘积，作为历史基线。

```powershell
D:\Anaconda_envs\envs\pytorch\python.exe .\alfoil_dnm\train.py
```

默认结果：`runs1/controlled/xsdd_dnm_v1_cls/`。逐轮保存交叉熵 loss、Accuracy、
Macro-P/R/F1、时间和显存；训练结束保存固定测试集结果、混淆矩阵、逐图预测、
参数量与 batch=1 CPU/GPU 前向延迟。

本目录中的 `data.py`、`loss.py`、`metrics.py`、`infer.py` 等旧文件仍服务于早期
APSPC 目标检测实验；新的分类训练入口不读取 APSPC 的 YAML 或框标签。

# DNM-V2b：几何平均版

V2b 与 V2a 的数据、骨干、参数、检测头、损失和训练协议完全一致。前向公式的
结构差异是分支聚合使用 `exp(mean(log(gate)))`；两者的胞体阈值分别按各自理论
初始膜电位校准，使初始胞体均值都约为 0.5。几何平均会改变原始函数，因此必须
作为独立模型报告，不能描述为与论文乘积完全等价。

```powershell
D:\Anaconda_envs\envs\pytorch\python.exe .\alfoil_dnm_v2b\train.py
```

结果写入 `runs/controlled/dnm_v2b/`，同时保存模型内部 loss 和统一检测指标。

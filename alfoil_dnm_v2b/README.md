# DNM-V2b：log 域几何平均检测器

V2b 只把 V2a 的分支聚合改为 `exp(mean(log(gate)))`，减弱多项相乘导致的尺度
收缩。几何平均不是论文原乘积的数学等价形式，因此必须作为独立消融报告。

```powershell
D:\Anaconda_envs\envs\pytorch\python.exe .\alfoil_dnm_v2b\train.py
```

结果写入 `run2/controlled/dnm_v2b/`，其余检测协议与 V2a 相同。

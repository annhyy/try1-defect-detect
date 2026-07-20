# DNM-V2b：论文公式、log 域几何平均

V2b 仅把 V2a 的分支聚合改为 `exp(mean(log(gate)))`，减弱分支内输入数量带来的
乘积收缩。它并非论文原乘积的数学等价实现，因此作为独立消融模型报告。

```powershell
D:\Anaconda_envs\envs\pytorch\python.exe .\alfoil_dnm_v2b\train.py
```

默认读取 X-SDD 固定七分类划分，结果保存到
`runs1/controlled/xsdd_dnm_v2b_cls/`。

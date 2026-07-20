# DNM-V2a：论文公式、log 域精确乘积

V2a 与其他分类实验共享数据、224×224 输入、卷积骨干、优化器和评价器。分类头
包含论文突触参数 `w`、`theta`、正距离 `d` 以及分支强度 `v`。突触响应为
`sigmoid((w*x-theta)/d)`；这与公式分母指数前的负号完全一致。分支乘积使用
`exp(sum(log(gate)))`，只改变数值计算方式，不改变数学上的连乘。

```powershell
D:\Anaconda_envs\envs\pytorch\python.exe .\alfoil_dnm_v2a\train.py
```

默认读取 X-SDD 固定七分类划分，结果保存到
`runs1/controlled/xsdd_dnm_v2a_cls/`。

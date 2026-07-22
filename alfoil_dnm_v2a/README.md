# DNM-V2a：log 域精确乘积检测器

V2a 与其他内部检测模型共享 APSPC 数据、640x640 输入、卷积骨干、检测头、
目标分配、损失和评价器。融合块包含论文突触参数 `w`、`theta`、正距离 `d` 和
分支强度 `v`；分支使用 `exp(sum(log(gate)))` 计算与直接连乘数学等价的乘积。

```powershell
D:\Anaconda_envs\envs\pytorch\python.exe .\alfoil_dnm_v2a\train.py
```

结果写入 `run2/controlled/dnm_v2a/`，最优权重按验证集 mAP50-95 选择。

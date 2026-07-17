# DNM-V2a：论文忠实乘积版

该入口复用 `alfoil_dnm/train.py` 的受控训练协议。与 V1 相比，它补充了每个
突触的正距离参数 `d`、每个分支的正强度 `v`，并修正突触公式。分支内仍是
论文原始乘积，只改为 `exp(sum(log(gate)))` 进行数值稳定计算，数学结果不变。

```powershell
D:\Anaconda_envs\envs\pytorch\python.exe .\alfoil_dnm_v2a\train.py
```

默认使用 4 个分支、每分支 8 个输入，结果写入 `runs/controlled/dnm_v2a/`。
其中 `metrics.csv` 保留全部训练/验证 loss，`comparison_metrics.csv` 保存统一
P、R、mAP、时间和显存。

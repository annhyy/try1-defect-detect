# 普通卷积检测对照

该入口与 DNM-V2a/V2b 共用 APSPC 数据、卷积骨干、检测头、目标分配、损失和
评价器，只把树突融合块替换为 `1x1 -> 深度 3x3 -> 1x1` 普通卷积分支。隐藏宽度
按 DNM 参数预算估算，用于检验差异是否来自乘性树突结构。

```powershell
D:\Anaconda_envs\envs\pytorch\python.exe .\comparisons\conv_control\train.py
```

结果写入 `run2/controlled/conv_control/`。

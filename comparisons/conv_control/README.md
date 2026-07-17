# 普通卷积参数匹配对照

该模型与 DNM-V2a/V2b 共用同一骨干、检测头、损失、数据和训练设置，只把树突
变换替换为 `1x1 -> 深度3x3 -> 1x1` 普通卷积分支。瓶颈宽度由 DNM 的分支数
和输入数自动计算，使默认模型参数量尽可能接近，从而检验提升是否真正来自树突。

```powershell
D:\Anaconda_envs\envs\pytorch\python.exe .\comparisons\conv_control\train.py
```

结果写入 `runs/controlled/conv_control/`。

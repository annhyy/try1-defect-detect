# 普通神经网络分类头对照

该入口与 DNM-V2a/V2b 共用同一个轻量卷积骨干、全局池化、数据和优化协议，只把
树突分类头替换为 `Linear + LayerNorm + SiLU + Linear`。隐藏宽度根据 DNM 头参数量
自动估算，用来判断差异来自乘性树突结构还是普通参数规模。

```powershell
D:\Anaconda_envs\envs\pytorch\python.exe .\comparisons\conv_control\train.py
```

默认读取 X-SDD 固定七分类划分，结果保存到
`runs1/controlled/xsdd_conv_control_cls/`。

# X-SDD 分类中的 DNM 消融结构

## 共同部分

四个内部模型都接收 `[B,3,224,224]` 图像，使用相同的 `TinyBackbone` 提取
`[B,128,H,W]` 特征，再通过全局平均池化得到 `[B,128]`。实验只替换最后的分类头，
输出均为 `[B,7]` logits，并统一使用交叉熵。这样可以直接检验乘性树突头相对普通
参数匹配头是否有价值。

## 论文公式中的负号

论文突触写为：

\[
S_{ij}(x_i)=\frac{1}{1+\exp[-(w_{ij}x_i-\theta_{ij})/d_{ij}]}.
\]

指数前确实有负号。由于 `sigmoid(z)=1/(1+exp(-z))`，代码等价式是：

```python
torch.sigmoid((weight * x - threshold) / distance)
```

若在 `sigmoid` 参数前再手动加负号，会把单调方向反转。V1 则故意保留早期基础代码
的 `sigmoid(-k*(w*x-theta))`，用于展示旧实现与论文公式修正版的区别。

## 四个分类头

### DNM-V1

投影到少量突触输入后，按“突触 sigmoid → 分支内直接连乘 → 分支求和 → 胞体”
计算。它保留旧符号和直接 `prod`，可能受连乘缩小与梯度衰减影响，属于历史基线。

### DNM-V2a：log 域精确乘积

- 使用论文方向的突触公式；
- 每个突触学习 `w`、`theta` 与正距离 `d`；
- 每个分支学习正强度 `v`；
- 以 `exp(sum(log(gate)))` 计算分支乘积，与直接连乘数学等价；
- 默认 4 个分支、每分支 8 个投影特征（V1 历史基线默认每分支 4 个）。

log 域能减少数值下溢，但不能从数学上消除“多个 0--1 因子相乘使真实梯度变小”
的问题。因此 V2a 是数值稳定版，不等同于完全解决连乘优化问题。

### DNM-V2b：log 域几何平均

分支改为 `exp(mean(log(gate)))`。这会改变原函数，却能避免输出随突触数指数收缩。
V2a/V2b 其他部分相同，因此它们的差异直接检验连乘尺度是否为主要瓶颈。

### Conv-Control

使用 `Linear → LayerNorm → SiLU → Linear`。程序按 V2 头的参数预算估算隐藏宽度，
使对比尽量不是“参数更多所以更准”。精确参数量会写入各实验的
`experiment_config.json` 和 `test_metrics.json`，报告时应引用实际值。

## 输出与解释

```powershell
D:\Anaconda_envs\envs\pytorch\python.exe .\alfoil_dnm\train.py
D:\Anaconda_envs\envs\pytorch\python.exe .\alfoil_dnm_v2a\train.py
D:\Anaconda_envs\envs\pytorch\python.exe .\alfoil_dnm_v2b\train.py
D:\Anaconda_envs\envs\pytorch\python.exe .\comparisons\conv_control\train.py
```

结果依次保存到 `runs1/controlled/xsdd_dnm_v1_cls`、`xsdd_dnm_v2a_cls`、
`xsdd_dnm_v2b_cls` 和 `xsdd_conv_control_cls`。若 DNM 优于普通头，只能说明这个数据、骨干和训练协议下树突头
更有效；若不优于，也只说明当前实现/设置没有证明优势，不能由一次实验否定所有
树突结构。

当前版本用标准反向传播统一训练四组模型。SMS 属于后续优化算法消融，不能只给
DNM 换优化器后仍声称结构对比完全受控；若加入，应把“结构”和“优化算法”拆成两张表。

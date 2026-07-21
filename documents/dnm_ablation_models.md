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

## 下一阶段独立版本

新代码和新结果位于 `alfoil_dnm_next/`，不会覆盖上述历史实现或
`runs1/controlled/xsdd_dnm_v1_cls` 等旧目录。

### DNM-V2a-F4 与 DNM-V2b-F4

这两个版本只把 `branch_features` 从 8 改为 4，其他突触公式、LayerNorm 后的输入
sigmoid、正分支强度、胞体斜率/阈值和训练协议保持 V2 原样。V2a 仍为
`exp(sum(log(gate)))`，V2b 仍为 `exp(mean(log(gate)))`。默认输出分别为
`xsdd_dnm_v2a_f4_cls` 和 `xsdd_dnm_v2b_f4_cls`。因此它们可以直接回答“八项连乘是否是
V2 失败的主要原因”，不能把后续训练策略变化归因到这个消融。

### DNM-V1-Tuned

V1-Tuned 默认使用 4 个分支、每分支 4 项：

```text
128 维卷积特征 -> 16 维（4 个分支各自 4 维） -> LayerNorm -> 分支内四项 log 乘积
```

每个分支获得不同的投影特征；LayerNorm 后不再接前置 sigmoid。输出是原始多分类 logits：

```text
L_c = sum_b v_cb * B_cb + bias_c
```

其中 `v_cb` 可以为正或负，`bias_c` 是独立类别偏置，不再使用正胞体斜率与耦合阈值。
默认训练为 150 epoch、骨干/分类头学习率 `1e-3/3e-3`，余弦最低学习率为初始值的 1%。

### 诊断与检查点

三个新 DNM 入口每轮在终端打印验证集 `pred_count=[...]`，并把 train/val 数量写入
`metrics.csv` 和 `comparison_metrics.csv`。每个输出目录保存 `best_accuracy.pt`、
`best_macro_f1.pt`、兼容旧脚本的 `best.pt`（Accuracy 别名）以及 `last.pt`。最终测试默认
只评估验证 Accuracy 选出的权重；Macro-F1 权重只依据验证集选择并单独保存。

类别加权交叉熵通过 `--class-weighting balanced` 显式开启。为了保持第二组实验公平，
`train_conv_control_weighted.py` 对普通卷积控制使用同一套类别权重，输出到
`xsdd_conv_control_weighted_cls`，旧的 `xsdd_conv_control_cls` 不变。

## 输出与解释

```powershell
D:\Anaconda_envs\envs\pytorch\python.exe .\alfoil_dnm\train.py
D:\Anaconda_envs\envs\pytorch\python.exe .\alfoil_dnm_v2a\train.py
D:\Anaconda_envs\envs\pytorch\python.exe .\alfoil_dnm_v2b\train.py
D:\Anaconda_envs\envs\pytorch\python.exe .\comparisons\conv_control\train.py

# 新版独立入口
D:\Anaconda_envs\envs\pytorch\python.exe .\alfoil_dnm_next\train_v2a_f4.py
D:\Anaconda_envs\envs\pytorch\python.exe .\alfoil_dnm_next\train_v2b_f4.py
D:\Anaconda_envs\envs\pytorch\python.exe .\alfoil_dnm_next\train_v1_tuned.py
```

结果依次保存到 `runs1/controlled/xsdd_dnm_v1_cls`、`xsdd_dnm_v2a_cls`、
`xsdd_dnm_v2b_cls` 和 `xsdd_conv_control_cls`。若 DNM 优于普通头，只能说明这个数据、骨干和训练协议下树突头
更有效；若不优于，也只说明当前实现/设置没有证明优势，不能由一次实验否定所有
树突结构。

当前版本用标准反向传播统一训练四组模型。SMS 属于后续优化算法消融，不能只给
DNM 换优化器后仍声称结构对比完全受控；若加入，应把“结构”和“优化算法”拆成两张表。

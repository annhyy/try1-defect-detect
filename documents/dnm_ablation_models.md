# DNM-V2a、DNM-V2b 与普通卷积消融说明

## 实验目的

三组模型使用完全相同的 APSPC 划分、640×640 Letterbox 输入、卷积骨干、
单尺度检测头、目标分配、损失函数、优化器和训练轮数，只替换中间特征融合块。
该实验回答的是“树突计算是否比参数量接近的普通卷积更有效”，而不是直接宣称
某个模型优于所有 YOLO。

默认参数量已经由本地前向测试核验：

| 模型 | 参数量 | 与 V2a 的差值 |
|---|---:|---:|
| DNM-V2a | 500,607 | 0 |
| DNM-V2b | 500,607 | 0 |
| Conv-Control | 500,486 | -121（约 -0.024%） |

## 论文公式中的负号

论文突触写为：

\[
S_{ij}(x_i)=\frac{1}{1+\exp\left[-(w_{ij}x_i-\theta_{ij})/d_{ij}\right]}.
\]

指数里确实有负号。根据 `sigmoid(z)=1/(1+exp(-z))`，PyTorch 等价式是：

```python
torch.sigmoid((weight * x - threshold) / distance)
```

不能再写成 `torch.sigmoid(-(weight*x-threshold)/distance)`，否则整体单调方向反转。

## 三种融合块

### DNM-V2a：log 域精确乘积

- 投影特征经 GroupNorm 和 sigmoid 映射到 0--1；
- 每个突触学习有界权重 `w`、阈值 `theta` 和正距离 `d`；
- 每个分支学习正强度 `v`，膜层实现论文的 `u=sum(v_j*b_j)`；
- 分支计算 `exp(sum(log(gate)))`，与直接连乘数学等价；
- 默认 4 个分支、每分支 8 个突触输入。

### DNM-V2b：log 域几何平均

V2b 与 V2a 的结构公式只有分支聚合不同：

```python
exp(mean(log(gate)))
```

几何平均会改变原始函数，但避免输入项增多时输出和梯度按指数缩小。由于两种
聚合的天然数值尺度不同，胞体阈值分别初始化为各自理论膜电位均值，使两组初始
胞体响应都约为 0.5；阈值之后均正常学习。V2a 与 V2b 参数量完全一致，因此
两者差异可用于判断乘积尺度是否是主要优化障碍。

### Conv-Control：普通卷积

树突分支被 `1×1卷积 -> 深度3×3卷积 -> 1×1卷积` 替代。程序根据 DNM 的
分支数和输入数自动计算瓶颈宽度；默认宽度为 51，使总参数量与 V2 相差仅 121。
残差、拼接、融合卷积和检测头均与 V2 保持一致。

## 运行命令

```powershell
D:\Anaconda_envs\envs\pytorch\python.exe .\alfoil_dnm_v2a\train.py
D:\Anaconda_envs\envs\pytorch\python.exe .\alfoil_dnm_v2b\train.py
D:\Anaconda_envs\envs\pytorch\python.exe .\comparisons\conv_control\train.py
```

默认输出分别为：

- `runs/controlled/dnm_v2a/`
- `runs/controlled/dnm_v2b/`
- `runs/controlled/conv_control/`

每个目录中的 `metrics.csv` 保存 train/val 的 total、obj、box、cls loss 以及统一
检测指标；`comparison_metrics.csv` 只保存可横向比较的 P、R、mAP、时间和显存。

如果 RTX 4060 在每分支 8 个突触输入、batch=8 下出现显存不足，应当给三组同时改成
相同的 `--batch-size 4`，不能只降低某一个模型的批大小。

## 推理兼容

新 checkpoint 保存 `variant` 字段。`alfoil_dnm/infer.py` 会自动构建 V1、V2a、
V2b 或 Conv-Control；旧 checkpoint 没有该字段时默认按 V1 加载。

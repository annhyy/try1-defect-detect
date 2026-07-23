# 空间树突卷积检测版本

本目录验证此前讨论的想法：让树突运算直接承担局部空间特征提取，替换骨干中的
普通卷积，而不是只把 DNM 放在分类头或卷积骨干之后。

## 方案

当前实现里有两个不同概念，不能混在一起：

| 名称 | 当前值 | 含义 |
|---|---:|---|
| `patch_items` | 4 | 来自 2x2 空间窗口的四个局部位置 |
| `branches` | 4 | 每个输出通道内并行的树突分支数 |

也就是说，当前不是“2x2 的四个位置分别对应四个分支”。严格说法是：每个输出通道
有 4 个树突分支，每个分支都读取同一个 2x2 邻域的 4 个位置，并在分支内部做四项
乘积。因此每个输出通道、每个空间位置一共有 `4 branches x 4 patch_items = 16`
个突触门。

一个树突分支读取一个 2x2 邻域的四个位置：

```text
输入通道
  -> 1x1 通道投影（不聚合空间）
  -> 每个分支取得 2x2 的四个局部输入
  -> Synapse：四个 sigmoid 突触响应
  -> Dendritic：分支内四项相乘
  -> Membrane：多个分支求和
  -> Soma：sigmoid 胞体输出
  -> BatchNorm + SiLU
```

可写为：

```text
z(c,b,p) = sigmoid((w(c,b,p) * x(c,b,p) - q(c,b,p)) / d(c,b,p))
B(c,b)   = product(z(c,b,1:4))
Y(c)     = sigmoid(k(c) * (sum_b B(c,b) - qs(c)))
```

这里的 1x1 投影只混合通道，不读取相邻像素。被替换层中的空间聚合完全来自
2x2 树突乘积，因此不是“普通 3x3 卷积后再接一个 DNM”。

### 每个分支的作用

每个分支可以理解成一个独立的局部缺陷模板。分支内部四项相乘近似逻辑 AND，
要求 2x2 的四个局部条件同时满足；多个分支在膜层求和，近似允许多个模板以 OR
的方式共同激活同一个输出通道。

四个分支读取相同的四个空间坐标，但拥有各自独立的 1x1 通道投影、突触权重、
阈值和距离，所以可以分别学习横向边缘、纵向裂纹、亮暗对比或斑点等不同模式。
一条分支也能把 2x2 映射成一个数，但只能表达一套乘性条件，容量更小。

把 `--branches` 改成 8 只会变成 `8 branches x 4 patch_items`，即增加并行的
2x2 模板数量，并不会扩大空间窗口。真正的大一号空间版本应另建 3x3 层，使
`patch_items=9`，也就是每个分支内部处理九个局部位置。

## 为什么使用 stride=1

2x2 树突窗口每次移动一个像素，右侧和下侧使用复制填充，输出高宽与输入相同。
下采样仍由骨干中的普通 stride=2 卷积完成。这样可以保留 80x80 的检测特征图，
避免小缺陷经过额外下采样后消失。

每个分支固定只乘四项。初始单分支输出约为 `0.5^4=0.0625`，不会采用几十或
几百项连乘。胞体阈值初始化为 `branches * 0.5^4`，使初始胞体输出位于约 0.5，
减少一开始就饱和的问题。

## 两个消融配置

先运行同目录提供的严格普通卷积对照。它与树突模型具有相同的层数、通道、下采样
位置和检测头，区别只在于候选替换位置仍使用普通 3x3 卷积：

```powershell
D:\Anaconda_envs\envs\pytorch\python.exe .\alfoil_dendritic_conv\train.py --variant conv
```

结果保存到 `run2/controlled/dendritic_conv_control/`。这个对照比旧的
`comparisons/conv_control` 更适合回答“树突能否替换卷积”，因为旧对照针对的是
DNM 融合块，结构并不与本实验一一对应。

`--replace-layers 1` 是默认配置，只替换骨干最后一个 128 通道、80x80 特征层：

```text
Conv s2 -> Conv -> Conv s2 -> Conv -> Conv s2 -> DendriticConv s1
```

严格展开为：

```text
640x640x3
  -> ConvBNAct 3->32, stride=2      = 320x320x32
  -> ConvBNAct 32->32, stride=1     = 320x320x32
  -> ConvBNAct 32->64, stride=2     = 160x160x64
  -> ConvBNAct 64->64, stride=1     = 160x160x64
  -> ConvBNAct 64->128, stride=2    = 80x80x128
  -> SpatialDendriticConv 128->128, 2x2, stride=1, branches=4
  -> ConvBNAct 检测 stem 128->128
  -> objectness / box / class heads
```

R1 总参数量为 361,583。被替换的 128 通道普通 3x3 卷积约 147,712 参数；对应的
2x2 空间树突层约 72,192 参数。

`--replace-layers 2` 还会替换 64 通道、160x160 处的卷积：

```text
Conv s2 -> Conv -> Conv s2 -> DendriticConv s1 -> Conv s2 -> DendriticConv s1
```

严格展开为：

```text
640x640x3
  -> ConvBNAct 3->32, stride=2      = 320x320x32
  -> ConvBNAct 32->32, stride=1     = 320x320x32
  -> ConvBNAct 32->64, stride=2     = 160x160x64
  -> SpatialDendriticConv 64->64, 2x2, stride=1, branches=4
  -> ConvBNAct 64->128, stride=2    = 80x80x128
  -> SpatialDendriticConv 128->128, 2x2, stride=1, branches=4
  -> ConvBNAct 检测 stem 128->128
  -> objectness / box / class heads
```

R2 总参数量为 344,303。它不是更大模型，而是把第二个普通卷积也替换成参数更少的
2x2 空间树突层。

先跑一层替换。只有一层替换能够稳定收敛并接近普通卷积时，才有必要运行两层版本。
两层版本的中间激活明显更大，GTX 1060 建议使用 batch 2 或 4。

## 训练命令

在项目根目录运行一层替换：

```powershell
D:\Anaconda_envs\envs\pytorch\python.exe .\alfoil_dendritic_conv\train.py
```

运行两层替换：

```powershell
D:\Anaconda_envs\envs\pytorch\python.exe .\alfoil_dendritic_conv\train.py --replace-layers 2 --batch-size 4
```

默认使用 APSPC 640x640 检测数据，结果分别保存在：

```text
run2/controlled/dendritic_conv_r1/
run2/controlled/dendritic_conv_r2/
run2/controlled/dendritic_conv_control/
```

保存内容与现有检测实验一致，包括逐轮检测损失、Precision、Recall、mAP50、
mAP50-95、耗时、显存、实验配置以及最佳和最终权重。此外，
`dendritic_stats.csv` 会逐轮保存每个替换层的突触权重、阈值、距离、胞体参数和
突触梯度范数，用于判断树突参数是否真正参与学习以及是否出现梯度消失。

## 如何判断是否值得继续

优先比较 `dendritic_conv_r1` 与普通卷积对照的测试集 mAP50-95、逐类 AP、峰值显存
和每张推理时间。若准确率下降且计算显著变慢，就没有必要继续增加替换层数；若一层
版本在小缺陷类别上有稳定收益，再运行两层版本判断收益是否来自更广泛的树突空间建模。

下一步若要扩大空间感受野，不应把 `branches` 从 4 改成 8 来解释。更干净的版本应是
单独新建 3x3 空间树突层：`patch_items=9`，每个分支内部处理九个局部位置，并独立
输出到 `run2/controlled/dendritic_conv_k3_r1/`。由于初始九项乘积约为
`0.5^9=0.00195`，比四项乘积小 32 倍，3x3 版本必须单独观察梯度、归一化和胞体
阈值，不能与当前 2x2 结果混为同一个消融。

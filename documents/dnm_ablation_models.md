# APSPC 检测中的 DNM 结构

本轮没有实现新的“树突替换卷积”结构，也没有把 V1-Tuned 分类头硬套到检测器。
检测代码使用仓库已有的 DNM-V1/V2a/V2b 和普通卷积对照，目的只是把任务从整图
分类恢复为框定位加类别预测。

## 共同检测部分

四个内部模型均使用 `TinyBackbone`，输出 stride-8 特征图；融合块之后接相同的
单尺度无锚检测头：1 个 objectness 通道、4 个边框通道和 10 个类别通道。目标按
中心点分配到网格，训练使用动态正负平衡 focal objectness、Smooth-L1 box loss 和
类别 BCE。最优权重按验证集 mAP50-95 选择。

## DNM-V1

卷积特征先由 1x1 投影为少量局部特征，然后执行：

```text
Synapse sigmoid -> branch product -> membrane sum -> soma sigmoid
```

DNM 输出与 1x1 残差分支拼接，再经融合卷积送入检测头。该结构保持历史实现，
不等同于空间 2x2/3x3 树突卷积。

## DNM-V2a

V2a 补全论文突触参数 `w/theta/d`、正分支强度和胞体斜率/阈值。分支乘积在 log 域
计算 `exp(sum(log(gate)))`，与直接乘积数学等价，只改善数值稳定性。

## DNM-V2b

V2b 仅把 V2a 的分支聚合改为 `exp(mean(log(gate)))`。几何平均改变了函数尺度，
用于检验多项连乘缩小是否是训练瓶颈；它不是论文原式。

## Conv-Control

使用 `1x1 -> depthwise 3x3 -> 1x1` 普通卷积分支替换 DNM 融合，骨干、残差、
融合层、检测头、数据和损失保持一致。默认参数量与 V2a/V2b 近似匹配。

这四组只能回答当前融合块在同一自定义检测器中的作用。它们与 YOLO 的多尺度头、
目标分配和损失均不同，因此不能用 loss 绝对值解释结构优劣，必须比较测试集 mAP。

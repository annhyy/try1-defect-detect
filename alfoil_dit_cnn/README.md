# Dit-CNN 铝材缺陷检测版本

本目录是独立于 `alfoil_dnm` 的新实验，不再使用
`Synapse -> Dendritic -> Membrane -> Soma` 乘性 DNM。它依据 NeurIPS 2024
论文 *Dendritic Integration Inspired Artificial Neural Networks Capture Data
Correlation*，在普通卷积中加入通道二次相关项：

```text
Y = Conv(X) + X^T A X
```

其中二次项只组合**同一空间位置的不同通道**，不组合相邻 2x2 像素，也不承担
下采样。该层固定为 `stride=1`，整个检测器仍由前面的普通卷积完成 8 倍下采样。

论文：<https://proceedings.neurips.cc/paper_files/paper/2024/hash/90b31ad371165eaac2dc6de8993fded7-Abstract-Conference.html>

官方实现：<https://github.com/liuchongming1999/Dendritic-integration-inspired-CNN-NeurIPS-2024>

## 当前结构

```text
640x640 RGB
  -> 普通卷积骨干，连续 3 次 stride=2
  -> 128x80x80 特征
  -> Dit 卷积：LayerNorm -> [普通 3x3 卷积 + 通道二次项] -> BN -> SiLU
  -> 普通 3x3 检测 stem
  -> objectness + box + class 检测头
```

只替换骨干最后一个 `stride=1` 卷积，符合论文“只在少数层加入二次神经元”以控制
成本的思路。二次矩阵 `A` 按官方代码从零初始化，因此训练刚开始时 Dit 层就是
普通卷积，后续才逐步学习通道相关性。

完整 128 通道二次矩阵需要 `128x128x128 = 2,097,152` 个二次参数，而且在
80x80 特征图上的计算量不适合 GTX 1060。默认先将二次分支投影到 16 通道，执行
精确的 `16x16x16` 二次积分后再投影回 128 通道。普通 3x3 卷积分支仍保持完整
128 通道。该工程化版本保留 Dit 的运算本质，同时显著减少显存和计算。

设置 `--quadratic-channels 128` 可移除投影，运行论文形式的完整通道二次矩阵；
该配置仅用于结构验证，不建议直接在 640 输入和 GTX 1060 上完整训练。

## 训练

在项目根目录运行：

```powershell
D:\Anaconda_envs\envs\pytorch\python.exe .\alfoil_dit_cnn\train.py
```

默认读取 `datasets/apspc_yolo_letterbox640/data.yaml`，使用 640x640 输入，训练
120 轮，结果写入：

```text
run2/controlled/dit_cnn/
```

输出包括 `metrics.csv`、`comparison_metrics.csv`、`experiment_config.json`、
`best.pt`、`last.pt` 和 `test_metrics.json`，指标为 Precision、Recall、mAP50 和
mAP50-95，与当前 DNM 和 YOLO 检测实验保持一致。

GTX 1060 若显存不足，可先使用：

```powershell
D:\Anaconda_envs\envs\pytorch\python.exe .\alfoil_dit_cnn\train.py --batch-size 4
```

## 这一版验证什么

该实验只回答一个问题：在相同的轻量检测框架内，用 Dit 的显式通道二次相关性
替换一个普通卷积，是否能提高 APSPC 的定位与分类指标。它不是原乘性 DNM 的延续，
也不能仅凭参数量或分类 Accuracy 证明有效，最终应以固定测试集 mAP50-95、逐类 AP、
显存和推理时间共同判断。

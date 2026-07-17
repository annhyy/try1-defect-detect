# 树突检测器、YOLO11 与 YOLO26：受控对比实验协议

## 目的

本文件规定三种模型的**严格可比较**训练方案：树突检测器、YOLO11n 和 YOLO26n。模型内部结构、检测头和损失函数本来就不同，因此不能比较 loss 的绝对值；论文的主比较指标应统一为 Precision、Recall、mAP50 和 mAP50-95。

## 共同控制变量

三组受控实验均使用以下设置：

| 项目 | 固定值 |
|---|---|
| 数据集 | `datasets/apspc_yolo_letterbox640/data.yaml` |
| 数据划分 | 同一份 train / val / test（随机种子 42） |
| 输入尺寸 | 640 × 640，等比例 Letterbox，不拉伸原图 |
| 训练轮数 | 120 |
| 批大小 | 8 |
| 随机种子 | 42 |
| 初始化 | 从零开始；不使用预训练权重 |
| 数据增强 | 关闭（颜色、翻转、Mosaic、MixUp 等均关闭） |
| 优化器 | AdamW，初始学习率 0.002，权重衰减 0.0001 |
| 学习率策略 | 余弦衰减至 0，无 warmup |
| AMP | 关闭 |
| 梯度累积 | 不使用；`nbs=8` |

模型的网络规模、检测头设计和损失公式属于待比较的模型差异，不能强行设成一致。树突模型使用树突乘积聚合；YOLO 使用各自的标准检测头。

## 运行命令

在项目根目录执行。三条命令都默认使用上述受控设置；`--device 0` 可按需显式指定第一张 NVIDIA 显卡。

```powershell
# 树突检测器
D:\Anaconda_envs\envs\pytorch\python.exe .\alfoil_dnm\train.py

# YOLO11n，从 yolo11n.yaml 随机初始化
D:\Anaconda_envs\envs\pytorch\python.exe .\comparisons\yolo11\train.py

# YOLO26n，从 yolo26n.yaml 随机初始化
D:\Anaconda_envs\envs\pytorch\python.exe .\comparisons\yolo26\train.py
```

若加入 `--pretrained`，该实验变为迁移学习实验，必须与从零训练的三组结果分开报告，不能混在同一张主对比表中。

## 统一输出与评价方法

每个实验输出在 `runs/controlled/` 下：

| 模型 | 目录 |
|---|---|
| 树突检测器 | `runs/controlled/dnm/` |
| YOLO11n | `runs/controlled/yolo11n/` |
| YOLO26n | `runs/controlled/yolo26n/` |

每组都必须保存：

- `comparison_metrics.csv`：统一字段 `epoch, precision, recall, map50, map50_95, epoch_seconds, elapsed_seconds, gpu_memory_mb, learning_rate`，用于横向曲线与表格。
- `experiment_config.json`：实际数据路径、随机种子、优化器、增强、预训练状态、参数量等可复现实验配置。
- `test_metrics.json`：以验证集 mAP50-95 最优权重在测试集上重新评价的最终指标。
- 最优权重 `best.pt` 与最后一轮权重 `last.pt`。YOLO 的原生逐轮日志另保留为 `results.csv`；树突模型的原生逐轮损失保留为 `metrics.csv`。

树突模型和 YOLO 都每轮计算以下统一指标：

- `Precision`：预测框中正确框的比例。
- `Recall`：真实缺陷框被检出的比例。
- `mAP50`：IoU=0.50 下，10 个类别 AP 的平均值。
- `mAP50-95`：IoU 从 0.50 到 0.95、步长 0.05 的 AP 平均值；这是最终主指标。

树突评估器使用与 YOLO 相同的类别、边框、NMS 与 IoU 定义，并在 0.50:0.95 的十个阈值上计算 AP。由于两类模型的 loss 定义不同，`train_total`、`box_loss`、`cls_loss` 等只用于观察**本模型自身是否收敛**，不参与模型优劣比较。

可以在三组训练完成后绘制统一指标曲线：

```powershell
D:\Anaconda_envs\envs\pytorch\python.exe .\comparisons\plot_metrics.py
```

图像输出为 `runs/controlled/metrics_comparison.png`，包含 Precision、Recall、mAP50 和 mAP50-95 四个子图，不绘制不可比的 loss。

## 树突模型的特别说明

APSPC 的训练图像约有 6,400 个 stride-8 网格位置，但平均每张图只有约 1.7 个缺陷框。旧版树突训练器的 objectness 正负样本权重过低，背景网格容易主导梯度，导致 objectness 和 mAP 长期偏低。

当前训练器将 objectness 改为动态正负样本平衡的 focal BCE，并按 `mAP50-95` 而非验证 loss 保存 `best.pt`。这是检测任务的必要修正，因此旧版树突训练结果不能与新的受控 YOLO 实验直接对比，需按本协议重新训练。

## 旧的非受控运行

`comparisons/yolo11/runs/apspc/` 若仍存在，属于此前 Ultralytics 默认训练实验：它可能包含默认增强、自动优化器或 AMP 等设置。该结果可作为参考，但**不能**列入本文件定义的严格受控主对比。新的受控结果只读取 `runs/controlled/`。

## 每轮终端输出

树突模型会打印自身的训练/验证损失，并在同一行打印统一指标和耗时，例如：

```text
epoch 012/120 train(total=..., obj=..., box=..., cls=...) val(total=..., obj=..., box=..., cls=...) P=... R=... mAP50=... mAP50-95=... time=... elapsed=...
```

YOLO 的 `box_loss`、`cls_loss`、`dfl_loss` 与树突模型的损失项没有一一对应关系；比较时只读取各自的 `comparison_metrics.csv`。

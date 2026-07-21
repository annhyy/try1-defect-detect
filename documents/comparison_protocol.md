# X-SDD 七分类统一对比实验协议

## 目的

比较 DNM-V1、DNM-V2a、DNM-V2b、参数匹配普通分类头、YOLO11n-cls 和
YOLO26n-cls。任务是判断整张图属于哪一种表面缺陷，不预测缺陷框。

正式结构对比的所有模型都从随机权重开始。YOLO 的 `--pretrained` 仅作为迁移学习
参考，必须单独成表，不能把预训练收益解释为 YOLO 结构收益。

本页下方的 100 epoch、统一 `1e-3` 学习率协议对应 V1/V2a/V2b/Conv-Control/YOLO
历史基线。`alfoil_dnm_next` 中的 V2a-F4/V2b-F4 继续遵守该协议；V1-Tuned 是明确
标记的第二阶段结构与训练联合版本，采用 150 epoch、骨干/分类头 `1e-3/3e-3` 和
1% 最低学习率，不能把它与 F4 的差异解释成单一结构消融。

## 固定控制变量

| 项目 | 固定值 |
|---|---|
| 数据 | `datasets/xsdd_yolo11_classification` |
| 划分 | train 946 / val 201 / test 204；seed=42；逐类分层；排除 9 张完全重复图 |
| 输入 | 224×224 RGB；像素缩放到 0--1 |
| epoch | 100 |
| batch | 64 |
| 优化器 | AdamW，lr=0.001，weight decay=0.0001 |
| 调度 | 余弦衰减至 0，无 warmup |
| AMP | 关闭 |
| 轻量增强 | 90%--100% 随机裁取、水平/垂直翻转、轻量亮度/对比度 |
| 模型选择 | 验证集 Top-1 Accuracy 最佳权重 |
| 最终评价 | 固定 test 集只在训练完成后评估一次 |

X-SDD 绝大多数原图为 128×128，训练时放大到 224×224。原图没有新增信息，但统一
尺寸使所有模型接口和计算量口径一致；验证和测试使用确定性 Resize + CenterCrop。

## 正式运行命令

```powershell
D:\Anaconda_envs\envs\pytorch\python.exe .\comparisons\yolo11\train.py
D:\Anaconda_envs\envs\pytorch\python.exe .\comparisons\yolo26\train.py
D:\Anaconda_envs\envs\pytorch\python.exe .\alfoil_dnm\train.py
D:\Anaconda_envs\envs\pytorch\python.exe .\alfoil_dnm_v2a\train.py
D:\Anaconda_envs\envs\pytorch\python.exe .\alfoil_dnm_v2b\train.py
D:\Anaconda_envs\envs\pytorch\python.exe .\comparisons\conv_control\train.py
```

所有入口默认随机初始化。若改动 img-size、batch、epoch、增强或随机种子，必须对全部
模型做相同修改。YOLO 的 `--fraction` 仅供冒烟测试，正式实验必须为 1.0。

## 结果目录

| 模型 | 输出 |
|---|---|
| DNM-V1 | `runs1/controlled/xsdd_dnm_v1_cls/` |
| DNM-V2a | `runs1/controlled/xsdd_dnm_v2a_cls/` |
| DNM-V2b | `runs1/controlled/xsdd_dnm_v2b_cls/` |
| 普通头 | `runs1/controlled/xsdd_conv_control_cls/` |
| YOLO11 | `runs1/controlled/xsdd_yolo11n_cls_scratch/` |
| YOLO26 | `runs1/controlled/xsdd_yolo26n_cls_scratch/` |

## 可以横向比较的指标

- Accuracy：全部测试图中分类正确比例；
- Macro-Precision / Macro-Recall / Macro-F1：逐类等权平均，减少类别不均衡影响；
- 参数量：可训练/总参数个数；
- 推理速度：同一机器 batch=1、纯模型前向，分别报告 GPU/CPU median 和 p95；
- 训练时间与峰值显存：训练成本的辅助指标。

交叉熵 loss 可用于观察收敛，但不同结构的 loss 数值不能直接当成最终优劣。最终判断
以独立 test 的 Accuracy、Macro-F1、参数量和同机推理速度为准。

## 可复现性边界

X-SDD 测试集只有 204 张，一张图约等于 0.49 个百分点；YOLO11 当前只错 3 张，单次
实验容易受随机种子影响。正式报告至少运行 seed 42、43、44，报告均值±标准差，不能
只挑最高一次。测试集不能用于调参，模型结构和超参数选择只看 train/val。

# APSPC 目标检测统一对比协议

## 任务

当前任务是 10 类目标检测：模型必须同时输出缺陷位置、类别和置信度。整图
Accuracy、Macro-F1 等分类指标不再作为主结果。旧 X-SDD 分类代码和 `runs1/`
结果只作历史记录。

内部机制对比包括 DNM-V1、DNM-V2a、DNM-V2b、参数匹配普通卷积，以及只在骨干
最后一个 stride=1 卷积中加入通道二次积分的 Dit-CNN；外部基线为 YOLO11n、
YOLO11s 和可选的 YOLO26n。YOLO11s 是本轮新增的大参数对照，参数量约 9.4M，
YOLO11n 约 2.6M。

## 数据和尺寸

| 项目 | 固定值 |
|---|---|
| 数据 | `datasets/apspc_yolo_letterbox640/data.yaml` |
| 划分 | train 1320 / val 376 / test 189，seed 42 |
| 类别 | 10 类 |
| 输入 | 640x640，等比例 letterbox，不拉伸 |
| 原始图像 | 通常 2560x1920 |

640 缓存中的图片和框已经一起变换。将它再次放大到 960 只会插值，不会恢复细节，
代码会拒绝这种组合。若以后测试 960，YOLO 必须改读
`datasets/apspc_yolo/data.yaml`，从原图只缩放一次；960 是独立实验，不能与 640
结果混成同一组消融。

## 默认训练设置

| 项目 | 默认值 |
|---|---|
| epochs | 120 |
| batch | 8；GTX 1060 跑 YOLO11s 可降为 4 |
| seed | 42 |
| 初始化 | 随机初始化 |
| 优化器 | AdamW，lr 0.002，weight decay 0.0001 |
| 调度 | 余弦衰减到 0，无 warmup |
| 数据增强 | 关闭，保持旧受控协议 |
| AMP | 关闭 |
| 最优权重 | 验证集 mAP50-95 最高 |
| 最终测试 | 训练结束后用 best 权重评估 test 一次 |

`--pretrained` 是单独的 COCO 迁移学习实验，不能与随机初始化的 DNM 直接解释为
结构优劣。若 YOLO11s 因显存改为 batch 4，严格比较 YOLO11n/YOLO11s 时也应把
YOLO11n 改成 batch 4 后重跑。

## 命令

```powershell
D:\Anaconda_envs\envs\pytorch\python.exe .\alfoil_dnm\train.py
D:\Anaconda_envs\envs\pytorch\python.exe .\alfoil_dnm_v2a\train.py
D:\Anaconda_envs\envs\pytorch\python.exe .\alfoil_dnm_v2b\train.py
D:\Anaconda_envs\envs\pytorch\python.exe .\comparisons\conv_control\train.py
D:\Anaconda_envs\envs\pytorch\python.exe .\alfoil_dit_cnn\train.py
D:\Anaconda_envs\envs\pytorch\python.exe .\comparisons\yolo11\train.py
D:\Anaconda_envs\envs\pytorch\python.exe .\comparisons\yolo11\train_s.py
D:\Anaconda_envs\envs\pytorch\python.exe .\comparisons\yolo26\train.py
```

## 保存内容

所有新检测运行放在 `run2/controlled/`，不覆盖 `runs/` 和 `runs1/` 的旧结果。

- DNM 和 Dit-CNN 的 `metrics.csv`：train/val total、objectness、box、class loss，
  以及 P/R/mAP。
- YOLO 的 `results.csv`：train/val box、class、DFL loss，以及 P/R/mAP。
- `comparison_metrics.csv`：统一的 epoch、Precision、Recall、mAP50、mAP50-95、
  每轮时间、累计时间、峰值显存和学习率。
- `experiment_config.json`：完整参数、数据路径、初始化方式和参数量。
- `test_metrics.json`：验证 mAP50-95 最优权重在固定测试集上的结果。
- `best.pt`、`last.pt` 或 YOLO 的 `weights/` 权重。

不同检测器的损失定义不同。DNM 的 objectness/box loss 和 YOLO 的 box/DFL loss
只能用于判断各自是否收敛，不能比较绝对值。模型优劣以同一测试集的 Precision、
Recall、mAP50、mAP50-95、参数量和同机速度为主。

## 绘图

```powershell
D:\Anaconda_envs\envs\pytorch\python.exe .\comparisons\plot_metrics.py
```

输出 `run2/controlled/metrics_comparison.png` 和
`run2/controlled/final_detection_comparison.png`。图中只混合可比较的框级指标，
不把不同模型的 loss 画成同一尺度。

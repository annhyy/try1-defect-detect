# APSPC 铝材表面缺陷目标检测

当前主任务是**目标检测**，不是整图分类。所有模型都读取同一份 APSPC YOLO
边框标注，同时预测缺陷位置、类别和置信度。

APSPC 共 1,885 张图、3,143 个缺陷框、10 个类别。本地固定划分为
train/val/test = 1320/376/189。数据集、权重和运行日志不提交到 Git。

## 输入尺寸

正式对比使用 `datasets/apspc_yolo_letterbox640/data.yaml` 和 640x640 输入。
这份缓存已经按原始比例缩放并填充，边框也同步变换，因此不会把 4:3 原图拉伸。

不要把 640 缓存再次放大到 960：二次插值不会产生新细节。以后若单独测试 960，
YOLO 必须改读 `datasets/apspc_yolo/data.yaml`，从 2560x1920 原图只缩放一次。

## 检测实验

在项目根目录运行：

```powershell
# 现有 DNM 检测器及历史消融
D:\Anaconda_envs\envs\pytorch\python.exe .\alfoil_dnm\train.py
D:\Anaconda_envs\envs\pytorch\python.exe .\alfoil_dnm_v2a\train.py
D:\Anaconda_envs\envs\pytorch\python.exe .\alfoil_dnm_v2b\train.py
D:\Anaconda_envs\envs\pytorch\python.exe .\comparisons\conv_control\train.py

# Dit-CNN 通道二次积分检测器
D:\Anaconda_envs\envs\pytorch\python.exe .\alfoil_dit_cnn\train.py

# 用 2x2 空间树突运算替换一个或两个普通卷积
D:\Anaconda_envs\envs\pytorch\python.exe .\alfoil_dendritic_conv\train.py --variant conv
D:\Anaconda_envs\envs\pytorch\python.exe .\alfoil_dendritic_conv\train.py
D:\Anaconda_envs\envs\pytorch\python.exe .\alfoil_dendritic_conv\train.py --replace-layers 2 --batch-size 4

# YOLO 目标检测对照
D:\Anaconda_envs\envs\pytorch\python.exe .\comparisons\yolo11\train.py
D:\Anaconda_envs\envs\pytorch\python.exe .\comparisons\yolo11\train_s.py
D:\Anaconda_envs\envs\pytorch\python.exe .\comparisons\yolo26\train.py

# 绘制检测指标
D:\Anaconda_envs\envs\pytorch\python.exe .\comparisons\plot_metrics.py
```

`YOLO11n` 是轻量对照，适配 10 类检测头后约 260 万参数；新增的
`YOLO11s` 约 940 万参数，用于观察更大容量是否有效。两者默认从 YAML 随机初始化，
显式加入 `--pretrained` 才运行单独的 COCO 迁移学习实验。

GTX 1060 跑 YOLO11s 时可能需要 `--batch-size 4`。严格比较 YOLO11n 与
YOLO11s 时，两组必须使用相同 batch 重新训练。

## 输出目录和指标

本轮所有新结果统一写入 `run2/controlled/<模型名>/`，不会覆盖 `runs/` 和
`runs1/` 的旧结果：

- DNM 的 `metrics.csv`：train/val 的 total、objectness、box、class loss；
- YOLO 的 `results.csv`：train/val 的 box、class、DFL loss 和验证指标；
- `comparison_metrics.csv`：逐轮 Precision、Recall、mAP50、mAP50-95、耗时、
  峰值显存和学习率；
- `experiment_config.json`：实际数据路径、输入尺寸、初始化、优化器、随机种子和参数量；
- `test_metrics.json`：验证集 mAP50-95 最优权重在固定测试集上的结果；
- `best.pt`、`last.pt`，或 YOLO 的 `weights/best.pt`、`weights/last.pt`。

绘图脚本生成 `run2/controlled/metrics_comparison.png` 和
`final_detection_comparison.png`。DNM 与 YOLO 的损失定义不同，因此统一图只画
P/R/mAP，不把不同 loss 的绝对值混在一起。

旧 X-SDD 分类代码和 `runs1/` 结果继续保留，但不属于当前 APSPC 检测对比。

详细说明见 [APSPC 数据说明](documents/apspc_dataset.md)、
[检测对比协议](documents/comparison_protocol.md) 和
[DNM 检测结构](documents/dnm_ablation_models.md)。Dit-CNN 的结构、论文对应关系和
训练方法见 [Dit-CNN 独立说明](alfoil_dit_cnn/README.md)；直接替换空间卷积的
实验见 [空间树突卷积说明](alfoil_dendritic_conv/README.md)。

## Git 范围

源码和说明文档纳入 Git；`datasets/`、`runs/`、`runs1/`、`run2/`、权重和
本地 IDE 文件均由 `.gitignore` 排除。

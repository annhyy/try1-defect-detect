# 树突检测器、YOLO11 与 YOLO26 的对比协议

## 模型选择

- 树突检测器：`alfoil_dnm/train.py`，树突部分严格采用突触—树突乘积—膜层求和—胞体 S 形激活；分支内仅使用 4 个投影局部特征。
- YOLO11：`comparisons/yolo11/train.py`，使用 `yolo11n`。
- YOLO26：`comparisons/yolo26/train.py`，使用 `yolo26n`。

YOLO11 是成熟、广泛使用的 Ultralytics 检测基线；YOLO26 是更新一代模型，加入端到端、默认免 NMS 推理和更轻的检测头。二者均为主流 Ultralytics 生态中的实时检测模型，但论文报告时应明确版本与安装包版本。

官方资料：<https://docs.ultralytics.com/models/yolo11>、<https://docs.ultralytics.com/models/yolo26>。

## 公平设置

1. 三个模型使用同一 `datasets/apspc_yolo/data.yaml`、同一训练/验证/测试划分、`img-size=640`、`epochs=120`、随机种子 42。
2. 第一组为**从零训练**：YOLO 脚本不加 `--pretrained`；树突模型本身从零训练。
3. 第二组可单独报告**迁移学习**：YOLO 加 `--pretrained`，树突模型只有在获得相同来源的预训练骨干后才能算严格可比。
4. 报告 mAP@0.5、mAP@0.5:0.95（YOLO 可直接输出）、每类 AP、召回率、参数量、显存、单图推理延迟。当前树突脚本每 5 个 epoch 计算一次 mAP@0.5；后续可加入 COCO 风格多阈值 mAP。

## 每个 epoch 的输出含义

树突训练器会打印：

```text
train(total=..., obj=..., box=..., cls=...) val(total=..., obj=..., box=..., cls=...) mAP50=...
```

- `total`：总损失。
- `obj`：目标存在性损失。
- `box`：边框回归损失。
- `cls`：缺陷类别损失。
- `mAP50`：预测框与真值框 IoU 阈值为 0.5 时的平均精度均值；按 `--map-interval` 计算，未计算时显示 `--`。

此前截图中的 `train=...` 与 `val=...` 只有总训练损失和验证损失，不能解释为准确率或 mAP。

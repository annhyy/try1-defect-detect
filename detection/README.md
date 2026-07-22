# APSPC 检测训练公共实现

- `dnm_train.py`：DNM-V1、V2a、V2b 和普通卷积对照共用的框检测训练循环；
- `yolo_train.py`：YOLO11n 与 YOLO11s 共用的 Ultralytics 检测训练循环。

这些文件只负责训练协议、检测指标和结果保存，不改变 `alfoil_dnm/model.py` 与
`model_variants.py` 中已有的 DNM 结构。用户入口仍位于各模型原目录下。

默认读取 `datasets/apspc_yolo_letterbox640/data.yaml`，所有新结果写入
`run2/controlled/`。

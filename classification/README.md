# X-SDD 七分类共享实现

当前主数据由 `comparisons/yolo11/prepare_xsdd.py` 从 `datasets/X-SDD/datas` 去重并
固定划分，输出层级为：

```text
datasets/xsdd_yolo11_classification/
  train/<类别>/*.{jpg,png}
  val/<类别>/*.{jpg,png}
  test/<类别>/*.{jpg,png}
  dataset_info.json
  split_manifest.csv
```

例如 `train/slag inclusion/1001.jpg` 由 `ImageFolder` 转为一个数字标签；送进模型的
batch 只有图片张量和数字标签，模型无法读取文件名或目录。

- `data.py`：DNM 和普通对照共用的缩放、增强和 0--1 张量变换；
- `metrics.py`：统一 Accuracy、Macro-P/R/F1、混淆矩阵和 batch=1 推理测速；
- `models.py`：三个 DNM 分类头和参数量近似匹配的普通分类头；
- `prepare_neu_cls.py`：保留的旧 NEU 整理脚本，不参与当前 X-SDD 实验。

YOLO 使用 Ultralytics 自身数据加载器，但读取完全相同的 train/val/test 目录；训练
结束后也使用本项目的统一评价器复评。

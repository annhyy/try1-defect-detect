# X-SDD 数据集说明

## 来源与任务

X-SDD 全名为 **Xsteel Surface Defect Dataset**，用于热轧钢带表面缺陷图像分类。
本地原始数据位于 `datasets/X-SDD/`，其中自带 `readme.txt`；公开项目来源为作者仓库
[Fighter20092392/X-SDD-A-New-benchmark](https://github.com/Fighter20092392/X-SDD-A-New-benchmark)。

数据一共包含 7 类、原始 1360 张 RGB 图像：

| 类别 | 原始数量 |
|---|---:|
| finishing roll printing | 203 |
| iron sheet ash | 122 |
| oxide scale of plate system | 63 |
| oxide scale of temperature system | 203 |
| red iron | 397 |
| slag inclusion | 238 |
| surface scratch | 134 |

绝大多数图像为 128×128；少量图像的高度略有差异。当前任务是整图七分类，不需要
XML、YOLO TXT 或缺陷框。

## 去重与固定划分

准备脚本为 `comparisons/yolo11/prepare_xsdd.py`。它计算所有图片的 SHA-256，发现
`red iron` 中有 9 组完全相同的图片，每组保留文件名排序靠前的一张，其余 9 张只从
准备集排除，原始目录不被修改。没有发现跨类别完全重复图。

去重后共 1351 张，以 seed 42 在每一类内部随机划分：

| 集合 | 数量 | 用途 |
|---|---:|---|
| train | 946 | 参数学习 |
| val | 201 | 每轮评价和最佳权重选择 |
| test | 204 | 训练结束后一次性最终评价 |

输出目录为 `datasets/xsdd_yolo11_classification/`，结构如下：

```text
train/<类别>/<图片>
val/<类别>/<图片>
test/<类别>/<图片>
dataset_info.json
split_manifest.csv
```

`dataset_info.json` 记录类别数量、重复组和划分规则；`split_manifest.csv` 记录每张图的
集合、类别、原始相对路径和 SHA-256。检查结果显示 train/val/test 之间没有完全相同
图片。脚本优先使用硬链接，所以准备目录不会重复占用一份图片空间。

## 标签如何进入模型

原始文件夹名就是类别名。准备后，PyTorch/Ultralytics 按父目录的字典序把类别映射成
0--6 的数字标签。训练 batch 只包含图像张量和数字标签；模型无法读取文件名或目录，
因此类别文件夹不是“把答案写进图片”。

## 当前 YOLO11 基线

随机初始化的 `yolo11n-cls.yaml` 在固定 test 集上得到 98.53% Accuracy 和 97.41%
Macro-F1，即 204 张中错 3 张。第一轮验证 Accuracy 为 14.93%，说明模型不是从一开始
就知道标签；训练约 60 轮后才稳定收敛。

该结果比此前接近 100% 饱和的 NEU 划分更适合本项目，但数据仍较小且较容易。正式
结论至少运行 3 个随机种子并报告均值±标准差，调参只看 train/val，不反复查看 test。

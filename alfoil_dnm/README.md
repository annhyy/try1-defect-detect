# 铝箔表面缺陷：轻量树突检测器

> 项目的正式入口和 APSPC 实际路径见仓库根目录 `README.md`；本文件仅说明自定义铝箔数据应满足的 YOLO 目录与标签格式。

本实现面向 `缺陷位置 + 缺陷类别` 的检测任务，而不是整图分类。它以轻量卷积提取局部感受野特征，再用树突头完成分组的非线性交互：突触门控 -> 分支几何平均（log 域）-> 膜层聚合 -> 胞体输出。分支只处理通道组，不对整幅图像或全部特征直接连乘，因此避免传统 DNM 在高维输入上的数值下溢。

## 推荐数据集与使用顺序

1. **原型/预训练：天池“铝型材表面缺陷识别”数据集**。公开研究资料表明其来自真实铝材产线，包含 10 类：non-conducting、scratch、corner leak、orange peel、paint leak、jet flow、paint bubble、pit、miscellaneous、dirty point。原始比赛为分类标签；若使用已重标注版本（常称 APDDD）或自行标框，即可直接训练本检测器。
2. **迁移验证：GC10-DET / Metal Surface Defects**。其框标注包含 crease、inclusion、oil spot、rolled pit、waist folding、burr 等，与划痕、脏污、凹坑、折皱较接近；仅用于验证结构，不应当作为“铝箔产线准确率”。
3. **最终部署：自建铝箔数据集**。以同一相机、照明和卷速采集，标注 `normal`（仅作负样本）及实际的破洞、划痕、凹坑、脏污、蚊虫等类别。建议每类至少 300 个实例、按生产批次而非随机图片切分训练/验证/测试，保留无缺陷图像作为硬负样本。

下载后的目录需是标准 YOLO 检测格式：

```
dataset/
  images/{train,val,test}/xxx.jpg
  labels/{train,val,test}/xxx.txt
  data.yaml
```

每个标签行：`class_id x_center y_center width height`（归一化到 0--1）。`normal` 不应成为检测类别；无缺陷图像保留空的 `.txt` 文件。

`data.yaml` 示例：

```yaml
path: D:/datasets/alfoil
train: images/train
val: images/val
test: images/test
names: [hole, scratch, pit, stain, insect]
```

## 训练与推理

安装 PyTorch（按 CUDA 版本从 PyTorch 官网选择）以及 `pyyaml pillow numpy` 后：

```powershell
# APSPC：先在仓库根目录执行 prepare_apspc.py，再开始训练
python .\alfoil_dnm\train.py --data .\datasets\apspc_yolo\data.yaml --epochs 120 --img-size 640 --batch-size 8 --branches 4 --out .\runs\apspc_dnm

# source 必须替换为实际待检测图片；此处使用 APSPC 原图作示例
python .\alfoil_dnm\infer.py --weights .\runs\apspc_dnm\best.pt --source .\datasets\APSPC1\img0.jpg --data .\datasets\apspc_yolo\data.yaml --out .\runs\apspc_dnm\prediction_img0.jpg
```

GTX 1060（6 GB）建议从 `640, batch=4~8, branches=4, width=32` 起步；CPU 用 `512, batch=2`。若缺陷宽度小于原图 1/16，请先对长条产线图做重叠切片（例如 640x640、20% 重叠），再合并检测结果，避免下采样抹掉小缺陷。

## 产线标注协议（关键）

- 破洞、蚊虫、杂质：框住可见边界；划痕、折痕：框住完整连续缺陷，过长时按物理间断拆分。
- 脏污与油污若在工艺上无法稳定区分，合为 `stain`；先保证标注一致性，再考虑细分类。
- 每张图记录批次、卷号、相机与光照条件。测试集只放训练中从未出现过的批次。
- 同一图的多个缺陷均标注。本模型支持多框、多类别和空标签负样本。

## 与给定基础 DNM 的对应

给定 `DNM_Linear2` 的突触—树突—膜—胞体思路被保留；本项目改为：

- 输入来自 CNN 的每个空间特征点，保留视觉局部感受野；
- 通道按树突分支分组，而非所有维度相乘；
- `exp(mean(log(gate)))` 代替直接 `prod`，使梯度和尺度稳定；
- 胞体输出作为检测头的共享表征，分别接 objectness、类别和边框回归分支。

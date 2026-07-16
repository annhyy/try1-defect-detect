# APSPC 数据集与训练记录

## 数据集来源

APSPC 是 **Aluminum Profile Surface Detection Database**。它由 2018 年广东工业智造大数据创新大赛“铝型材表面瑕疵识别”的天池原始分类数据重标注而来，下载页说明其为缺陷检测数据集。

- 天池原始比赛页：<https://tianchi.aliyun.com/competition/entrance/231682/information>
- APSPC 下载说明：<https://www.cvmart.net/dataSets/detail/272>
- 当前本地原始数据：`datasets/APSPC1`、`datasets/APSPC2`、`datasets/APSPC-Annotations/Annotations`

## 标注格式：Pascal VOC XML

每张图片对应一个同名 XML 文件。Pascal VOC 是长期广泛使用的目标检测标注格式，常被 LabelImg、VOCdevkit、MMDetection 等工具支持；其类别和边框都写在 XML 内。它适合保存人工标注，但本项目训练器使用 YOLO TXT，因此需要转换。

典型结构：

```xml
<object>
  <name>budaodian</name>
  <bndbox>
    <xmin>1</xmin><ymin>835</ymin>
    <xmax>2560</xmax><ymax>1313</ymax>
  </bndbox>
</object>
```

`name` 是明确的缺陷类别；`bndbox` 是像素坐标边框。转换后的 YOLO 标签每行格式为：`class_id x_center y_center width height`，后四项归一化到 0--1。

## 本地核验（2026-07-16）

- 图像：1,885 张，尺寸通常为 2560 x 1920。
- XML：1,885 个，Python XML 解析全部通过；图像与 XML 文件名一一对应。
- 缺陷框：3,143 个；一张图可有多个缺陷框。
- 没有独立的无缺陷图片；因此该数据集适合多类检测预训练，但最终铝箔系统必须加入产线正常图作为负样本。

## 类别与实例统计

| ID | XML 标签 | 中文含义 | 实例数 |
|---:|---|---|---:|
| 0 | `aoxian` | 凹陷 | 156 |
| 1 | `budaodian` | 不导电 | 921 |
| 2 | `cahua` | 划伤 | 328 |
| 3 | `jupi` | 橘皮 | 226 |
| 4 | `loudi` | 漏底 | 116 |
| 5 | `pengshang` | 碰伤 | 58 |
| 6 | `qikeng` | 气坑 | 423 |
| 7 | `tucengkailie` | 涂层开裂 | 103 |
| 8 | `tufen` | 凸粉 | 174 |
| 9 | `zangdian` | 脏点 | 638 |

`pengshang`、`tucengkailie`、`loudi` 和 `aoxian` 样本较少。首轮训练将保留真实分布；后续根据每类 AP、召回率决定是否只对训练集做增强或重采样，验证与测试集不增强。

## 转换与训练

在项目根目录执行：

```powershell
D:\Anaconda_envs\envs\pytorch\python.exe .\alfoil_dnm\prepare_apspc.py
D:\Anaconda_envs\envs\pytorch\python.exe .\alfoil_dnm\train.py --data .\datasets\apspc_yolo\data.yaml --epochs 120 --img-size 640 --batch-size 8 --branches 4 --out .\runs\apspc_dnm
```

转换脚本使用固定随机种子 42，将原图硬链接到 `datasets/apspc_yolo`（若硬链接不支持则复制），以 70% / 20% / 10% 生成训练、验证、测试集，且不改动原始压缩包解压目录。训练输出的权重、日志和数据均不提交到 Git；代码、文档和配置纳入 Git 管理。

## 当前实验状态

此前完成的合成数据实验只验证模型与训练链路可运行，不能作为 APSPC 或铝箔产线精度。APSPC 转换完成后，应报告 mAP@0.5、每类 AP、召回率、推理延迟与误检/漏检样例；再用真实铝箔数据做迁移微调。

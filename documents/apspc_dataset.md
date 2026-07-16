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

## 本项目如何处理标签

训练器不会直接读取 XML。执行 `alfoil_dnm/prepare_apspc.py` 后，它按以下步骤把 APSPC 转为本项目使用的 YOLO 数据：

1. 遍历 `APSPC-Annotations/Annotations/*.xml`，用 Python 标准库解析 XML。
2. 从 `<filename>` 找到 `APSPC1` 或 `APSPC2` 中的同名 JPG；脚本若找不到或发现重名，会立即报错，避免图片—标签错配。
3. 从每个 `<object>` 读取 `name` 和 `xmin/ymin/xmax/ymax`；边框会被裁剪到图像边界，非法框被跳过。
4. 按本表 ID 将字符串类别转为数字，计算并写入 YOLO 行：

   ```text
   class_id center_x/image_width center_y/image_height box_width/image_width box_height/image_height
   ```

   例如 `budaodian`（ID 1）的像素框 `(1, 835, 2560, 1313)` 在 2560 x 1920 图片上会写成近似：`1 0.500195 0.559896 0.999609 0.248958`。
5. 使用种子 42 随机划分图片：70% 训练、20% 验证、10% 测试；同一张图及其全部缺陷框只会进入一个集合，避免数据泄漏。
6. 生成 `datasets/apspc_yolo/images/{train,val,test}`、`labels/{train,val,test}` 和 `data.yaml`。图像优先硬链接到新目录，失败才复制，原始文件不改动。

`alfoil_dnm/data.py` 只读取这种 YOLO 目录和 `data.yaml`。本项目正式训练命令应使用 `datasets/apspc_yolo/data.yaml`；未来用铝箔产线数据时，也应先整理为同样的 YOLO 目录结构。

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

### `datasets/apspc_yolo`：本地真实 APSPC 转换数据

该目录由 APSPC XML 转换而来，包含真实铝型材图片和 10 类缺陷标签。截至本文档更新时，APSPC 已完成转换，但尚未产生有效的完整训练权重；此前交互会话对前台长训练有 60 秒限制。

APSPC 原始图像与转换数据合计约 3.4 GB（转换版本与原图使用硬链接时并不额外占用等量磁盘），不适合普通 Git 推送；另外数据页许可证信息未明确。因此不上传到此仓库。使用者应自行从来源下载后执行转换脚本。

真实 APSPC 训练完成后，应报告 mAP@0.5、每类 AP、召回率、推理延迟与误检/漏检样例；再用真实铝箔数据做迁移微调。

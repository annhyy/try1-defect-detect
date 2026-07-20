# NEU 六类表面缺陷数据说明

## 当前本地数据

本地原始目录为 `datasets/NEU-CLS/`，共找到 1,800 张 200×200 RGB 图片，类别均衡，
每类 300 张。文件名采用 NEU 表面缺陷数据常见命名：

| 类别 | 中文含义（近似） | 原始图片数 |
|---|---|---:|
| `crazing` | 龟裂/裂纹状缺陷 | 300 |
| `inclusion` | 夹杂 | 300 |
| `patches` | 斑块 | 300 |
| `pitted_surface` | 麻点/点蚀表面 | 300 |
| `rolled-in_scale` | 轧入氧化皮 | 300 |
| `scratches` | 划痕 | 300 |

NEU Surface Defect Database 是公开研究中常用的热轧钢带六分类数据。当前下载包
还包含一图一 TXT 的 YOLO 检测框，说明它是经过二次目标框标注/重新打包的版本，
而不是只含类别目录的原始发布形态。本地包中没有附带来源 URL、版本号或许可证
文件，因此报告中只能把它准确描述为“本地下载的 NEU 六类重打包版本”；正式提交
前应补记实际下载页面和许可，不能由代码反推出来源。

## 分类标签到底来自哪里

本项目不使用 TXT 框。整理规则为：

1. `crazing_10.jpg` 的前缀唯一对应 `crazing`；其他五类同理。
2. 脚本把图片硬链接/复制到 `train/crazing/` 等类别文件夹。
3. PyTorch `ImageFolder` 按字母顺序把父目录映射成数字标签。
4. DataLoader 返回 `(图像张量, 数字标签)`；模型前向只有 `model(images)`，看不到
   文件名、路径或目录名，因此不存在“从文件名偷看答案”。

目录到数字标签的实际映射保存在训练输出的 `experiment_config.json`。当前为：

```text
0 crazing
1 inclusion
2 patches
3 pitted_surface
4 rolled-in_scale
5 scratches
```

## 数据检查与重新划分

下载包自带的 train/valid 是为二次打包任务准备的，实际检查为训练 1,770、验证 30，
验证集过小，不适合可靠比较。`classification/prepare_neu_cls.py` 合并全部图片后：

- 计算 SHA-256，发现 `patches_101.jpg` 与 `patches_105.jpg` 完全相同；排除后者，
  防止相同图跨集合造成数据泄漏；
- 以 seed 42 逐类打乱，按 70%/15%/15% 分层划分；
- 得到 train 1,259、val 270、test 270，总计 1,799；
- `patches` 训练集 209 张，其他类别训练集各 210 张；val/test 每类各 45 张；
- 原图不预先放大，训练加载时才变换到 224×224。

完整清单在 `datasets/neu_cls_classification/split_manifest.csv`，统计和重复图记录在
`dataset_info.json`。数据目录被 Git 忽略；换机器时需要先自行取得原始数据，再运行：

```powershell
D:\Anaconda_envs\envs\pytorch\python.exe .\classification\prepare_neu_cls.py
```

## 任务边界

这是钢材/金属表面整图分类数据，适合在截止时间内验证树突分类结构，也能先确认
数据与训练链路是否可学。它不能证明铝箔产线上破洞、蚊虫等目标的定位能力，也不能
输出框。若最终需求必须标出缺陷位置，仍需回到可靠框标注数据做目标检测实验。

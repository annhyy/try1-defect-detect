# 树突神经元铝材缺陷检测

面向铝箔/铝材产线的轻量目标检测项目：以卷积骨干提取局部感受野，再由稳定树突模块完成局部非线性交互，输出缺陷位置、类别和置信度。

仓库内置 `demo_alfoil`，是 150 张**合成**铝材缺陷图，用于确保克隆仓库后可直接训练、推理并检查输出。真实 APSPC 数据集不包含在 Git 仓库中：原始数据体积大、许可证信息未明确，需要从原始平台下载后本地转换。APSPC 含 1,885 张铝型材图像、10 个类别、3,143 个 Pascal VOC XML 标注框，可用于预训练与方法验证；最终部署前必须用铝箔产线图像对破洞、蚊虫等类别进行微调。

## 快速开始

环境：Python 3.10+、PyTorch、PyYAML、Pillow、NumPy。使用本地 Anaconda 环境：

```powershell
D:\Anaconda_envs\envs\pytorch\python.exe -m pip install -r .\alfoil_dnm\requirements.txt
```

### 1. 直接运行内置 Demo（无需另行下载数据）

```powershell
# 训练：使用仓库内置的合成数据；训练输出保存到 runs/demo_dnm
D:\Anaconda_envs\envs\pytorch\python.exe .\alfoil_dnm\train.py --data .\demo_alfoil\data.yaml --epochs 30 --img-size 256 --batch-size 16 --width 16 --branches 4 --workers 0 --out .\runs\demo_dnm

# 推理：使用训练产生的 best.pt 和一张验证图片
D:\Anaconda_envs\envs\pytorch\python.exe .\alfoil_dnm\infer.py --weights .\runs\demo_dnm\best.pt --source .\demo_alfoil\images\val\demo_0123.jpg --data .\demo_alfoil\data.yaml --conf 0.5 --out .\runs\demo_dnm\prediction_demo_0123.jpg
```

### 2. 使用真实 APSPC 数据

解压 APSPC 原始压缩包到 `datasets/APSPC1`、`datasets/APSPC2` 和 `datasets/APSPC-Annotations` 后：

```powershell
# XML (Pascal VOC) -> YOLO 标签；原始数据不会被修改
D:\Anaconda_envs\envs\pytorch\python.exe .\alfoil_dnm\prepare_apspc.py

# 训练轻量树突检测器
D:\Anaconda_envs\envs\pytorch\python.exe .\alfoil_dnm\train.py --data .\datasets\apspc_yolo\data.yaml --epochs 120 --img-size 640 --batch-size 8 --branches 4 --out .\runs\apspc_dnm

# 推理
D:\Anaconda_envs\envs\pytorch\python.exe .\alfoil_dnm\infer.py --weights .\runs\apspc_dnm\best.pt --source sample.jpg --data .\datasets\apspc_yolo\data.yaml --out prediction.jpg
```

## 项目结构

```text
alfoil_dnm/       模型、VOC 转换、训练和推理代码
documents/        数据来源、类别和实验记录
datasets/         本地数据集（已忽略，不上传 Git）
demo_alfoil/      已提交到仓库的合成可运行示例数据
runs/             训练权重和预测结果（已忽略，不上传 Git）
```

详细的数据来源、类别映射、标注统计与实验规范见 [APSPC 数据说明](documents/apspc_dataset.md)。

## Git 约定

源码、文档、配置和 `demo_alfoil` 提交到仓库；真实数据集、权重、运行日志、IDE 缓存由 `.gitignore` 排除。每次真实实验完成后，在 `documents/apspc_dataset.md` 中记录数据版本、超参数、硬件、mAP、各类召回率和失败样例，再提交对应代码与记录。

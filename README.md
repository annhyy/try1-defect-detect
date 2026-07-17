# 树突神经元铝材缺陷检测

面向铝箔/铝材产线的轻量目标检测项目：以卷积骨干提取局部感受野，再由稳定树突模块完成局部非线性交互，输出缺陷位置、类别和置信度。

真实训练数据为 **APSPC（Aluminum Profile Surface Detection Database）**：1,885 张铝型材图像、10 个类别、3,143 个 Pascal VOC XML 标注框。它不包含在 Git 仓库中：原始数据体积大、许可证信息未明确，需要从原始平台下载后本地转换。APSPC 可用于预训练与方法验证；最终部署前必须用铝箔产线图像对破洞、蚊虫等类别进行微调。

## 快速开始

环境：Python 3.10+、PyTorch、PyYAML、Pillow、NumPy。使用本地 Anaconda 环境：

```powershell
D:\Anaconda_envs\envs\pytorch\python.exe -m pip install -r .\alfoil_dnm\requirements.txt
```

解压 APSPC 原始压缩包到 `datasets/APSPC1`、`datasets/APSPC2` 和 `datasets/APSPC-Annotations` 后：

```powershell
# XML (Pascal VOC) -> YOLO 标签；原始数据不会被修改
D:\Anaconda_envs\envs\pytorch\python.exe .\alfoil_dnm\prepare_apspc.py

# 等比例缩放为 640×640 letterbox 缓存（推荐，只需执行一次）
D:\Anaconda_envs\envs\pytorch\python.exe .\alfoil_dnm\cache_letterbox.py

# 机制消融：V1、论文乘积 V2a、几何平均 V2b、参数匹配普通卷积
D:\Anaconda_envs\envs\pytorch\python.exe .\alfoil_dnm\train.py
D:\Anaconda_envs\envs\pytorch\python.exe .\alfoil_dnm_v2a\train.py
D:\Anaconda_envs\envs\pytorch\python.exe .\alfoil_dnm_v2b\train.py
D:\Anaconda_envs\envs\pytorch\python.exe .\comparisons\conv_control\train.py

# 外部基线：YOLO11、YOLO26，从零训练
D:\Anaconda_envs\envs\pytorch\python.exe .\comparisons\yolo11\train.py
D:\Anaconda_envs\envs\pytorch\python.exe .\comparisons\yolo26\train.py

# 已完成的模型会被自动加入 Precision、Recall、mAP50、mAP50-95 对照图
D:\Anaconda_envs\envs\pytorch\python.exe .\comparisons\plot_metrics.py

# 推理
# 将 --source 替换为待检测图片；下面以本地 APSPC 原始图片为例
D:\Anaconda_envs\envs\pytorch\python.exe .\alfoil_dnm\infer.py --weights .\runs\controlled\dnm\best.pt --source .\datasets\APSPC1\img0.jpg --data .\datasets\apspc_yolo_letterbox640\data.yaml --out .\runs\controlled\dnm\prediction_img0.jpg
```

## 项目结构

```text
alfoil_dnm/       模型、VOC 转换、训练和推理代码
alfoil_dnm_v2a/   论文参数补全、log 域精确乘积入口
alfoil_dnm_v2b/   几何平均乘性树突入口
comparisons/      参数匹配普通卷积与 YOLO 外部基线
documents/        数据来源、类别和实验记录
datasets/         本地数据集（已忽略，不上传 Git）
runs/             训练权重和预测结果（已忽略，不上传 Git）
```

详细的数据来源、类别映射、标注统计与实验规范见 [APSPC 数据说明](documents/apspc_dataset.md)。

树突模型、YOLO11 和 YOLO26 的统一对比设置、评价指标和命令见[对比实验协议](documents/comparison_protocol.md)。

V2a、V2b 与普通卷积的公式、参数匹配方式和消融边界见[DNM 消融模型说明](documents/dnm_ablation_models.md)。

## Git 约定

源码、文档和配置提交到仓库；真实数据集、权重、运行日志、IDE 缓存由 `.gitignore` 排除。每次真实实验完成后，在 `documents/apspc_dataset.md` 中记录数据版本、超参数、硬件、mAP、各类召回率和失败样例，再提交对应代码与记录。

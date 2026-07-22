# YOLO26n APSPC 检测基线

该可选历史基线现在运行 APSPC 目标检测，不再运行 X-SDD 整图分类。数据、640 输入
和受控训练设置与 YOLO11 一致。

```powershell
D:\Anaconda_envs\envs\pytorch\python.exe .\comparisons\yolo26\train.py
```

默认随机初始化；加入 `--pretrained` 后作为单独的迁移学习实验。结果写入
`run2/controlled/yolo26n/`。

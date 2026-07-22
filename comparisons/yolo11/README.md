# YOLO11 APSPC 检测对照

两个入口都使用相同的 APSPC train/val/test 划分和 640x640 letterbox 缓存：

```powershell
# YOLO11n，适配 10 类检测头后约 260 万参数
D:\Anaconda_envs\envs\pytorch\python.exe .\comparisons\yolo11\train.py

# YOLO11s，约 940 万参数，本轮新增的大参数对照
D:\Anaconda_envs\envs\pytorch\python.exe .\comparisons\yolo11\train_s.py
```

两组默认从 YAML 随机初始化。加入 `--pretrained` 才运行单独的 COCO 迁移学习实验。
GTX 1060 跑 YOLO11s 可能需要 `--batch-size 4`；比较 n/s 时应给两组设置相同 batch。

结果分别写入 `run2/controlled/yolo11n/` 和 `run2/controlled/yolo11s/`。
Ultralytics 的 `results.csv` 保存 box/class/DFL loss，项目另外生成统一的
`comparison_metrics.csv` 和测试集 `test_metrics.json`。

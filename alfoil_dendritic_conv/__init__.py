"""使用空间树突运算替换普通卷积的 APSPC 检测实验。"""

from .model import DendriticConvDetector, PlainConvDetector, SpatialDendriticConv2d

__all__ = ["DendriticConvDetector", "PlainConvDetector", "SpatialDendriticConv2d"]

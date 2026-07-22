"""用于 APSPC 缺陷检测的 Dit-CNN 实验模型。"""

from .model import ChannelQuadratic, DitConvBlock, DitDetector

__all__ = ["ChannelQuadratic", "DitConvBlock", "DitDetector"]

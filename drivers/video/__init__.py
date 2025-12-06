"""
视频驱动插件
"""
from .base import VideoDriver
from .null import NullVideoDriver

__all__ = [
    "VideoDriver",
    "NullVideoDriver",
]

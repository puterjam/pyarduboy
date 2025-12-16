"""
空视频驱动 - 不显示任何内容，用于无头模式或测试
"""
import numpy as np
from .base import VideoDriver


class NullVideoDriver(VideoDriver):
    """空视频驱动，不输出任何显示"""

    def init(self, width: int, height: int) -> bool:
        self._width = width
        self._height = height
        self._running = True
        return True

    def render(self, frame_buffer: np.ndarray) -> None:
        # 不做任何事情
        pass

    def close(self) -> None:
        self._running = False

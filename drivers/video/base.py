"""
视频驱动基类，继承自 drivers.base.VideoDriver
"""
from ..base import VideoDriver as BaseVideoDriver


class VideoDriver(BaseVideoDriver):
    """视频驱动基类，提供一些通用工具方法"""

    def __init__(self):
        self._width = 0
        self._height = 0
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

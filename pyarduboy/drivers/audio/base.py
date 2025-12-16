"""
音频驱动基类
"""
from ..base import AudioDriver as BaseAudioDriver


class AudioDriver(BaseAudioDriver):
    """音频驱动基类，提供一些通用工具方法"""

    def __init__(self):
        self._sample_rate = 0
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

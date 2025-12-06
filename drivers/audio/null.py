"""
空音频驱动 - 不播放任何声音，用于无声模式或测试
"""
import numpy as np
from .base import AudioDriver


class NullAudioDriver(AudioDriver):
    """
    空音频驱动，不输出任何声音

    用于无声模式或测试场景

    Args:
        sample_rate: 采样率，默认 50000 Hz (Ardens 标准: 16MHz / 320 = 50kHz)
    """

    def init(self, sample_rate: int = 50000) -> bool:
        """
        初始化空音频驱动

        Args:
            sample_rate: 采样率，默认 50000 Hz (Ardens 标准)

        Returns:
            总是返回 True
        """
        self._sample_rate = sample_rate
        self._running = True
        return True

    def play_samples(self, samples: np.ndarray) -> None:
        # 不做任何事情
        pass

    def close(self) -> None:
        self._running = False

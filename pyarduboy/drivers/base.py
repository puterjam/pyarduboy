"""
PyArduboy 驱动基类定义

定义了视频、音频、输入驱动的抽象接口，所有驱动插件必须实现这些接口
"""
from abc import ABC, abstractmethod
from typing import Optional, Tuple
import numpy as np


class VideoDriver(ABC):
    """视频驱动抽象基类"""

    @abstractmethod
    def init(self, width: int, height: int) -> bool:
        """
        初始化显示设备

        Args:
            width: 显示宽度（像素）
            height: 显示高度（像素）

        Returns:
            初始化成功返回 True，失败返回 False
        """
        pass

    @abstractmethod
    def render(self, frame_buffer: np.ndarray) -> None:
        """
        渲染一帧到显示设备

        Args:
            frame_buffer: 帧缓冲区，numpy 数组格式
                         shape 为 (height, width, 3) RGB 格式
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """关闭显示设备，释放资源"""
        pass

    @property
    @abstractmethod
    def is_running(self) -> bool:
        """检查驱动是否正在运行"""
        pass


class AudioDriver(ABC):
    """音频驱动抽象基类"""

    @abstractmethod
    def init(self, sample_rate: int = 44100) -> bool:
        """
        初始化音频设备

        Args:
            sample_rate: 采样率，默认 44100 Hz

        Returns:
            初始化成功返回 True，失败返回 False
        """
        pass

    @abstractmethod
    def play_samples(self, samples: np.ndarray) -> None:
        """
        播放音频采样

        Args:
            samples: 音频采样数据，numpy 数组格式
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """关闭音频设备，释放资源"""
        pass


class InputDriver(ABC):
    """输入驱动抽象基类"""

    # Arduboy 按键映射（libretro 标准）
    BUTTON_UP = 4
    BUTTON_DOWN = 5
    BUTTON_LEFT = 6
    BUTTON_RIGHT = 7
    BUTTON_A = 8
    BUTTON_B = 0

    @abstractmethod
    def init(self) -> bool:
        """
        初始化输入设备

        Returns:
            初始化成功返回 True，失败返回 False
        """
        pass

    @abstractmethod
    def poll(self) -> dict:
        """
        轮询输入状态

        Returns:
            返回按键状态字典，格式：
            {
                'up': bool,
                'down': bool,
                'left': bool,
                'right': bool,
                'a': bool,
                'b': bool,
            }
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """关闭输入设备，释放资源"""
        pass

    @property
    @abstractmethod
    def is_running(self) -> bool:
        """检查驱动是否正在运行"""
        pass

"""
Pygame Mixer 音频驱动（直通模式）
"""
import numpy as np
from typing import Optional
from .base import AudioDriver

try:
    import pygame.mixer
    import pygame.sndarray
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False


class PygameMixerDriver(AudioDriver):
    """使用 pygame.mixer 播放原始 int16 音频，不做音量或重采样处理。"""

    def __init__(
        self,
        sample_rate: Optional[int] = None,
        channels: int = 2,
        buffer_size: int = 256
    ):
        super().__init__()

        if not PYGAME_AVAILABLE:
            raise ImportError("pygame is not installed. Please run: pip install pygame")

        self._sample_rate = sample_rate or 0
        self.channels = channels
        self.buffer_size = buffer_size

        self._initialized = False
        self._channel = None

    def init(self, sample_rate: int = 50000) -> bool:
        """
        初始化音频驱动

        Args:
            sample_rate: 采样率，默认 50000 Hz (Ardens 标准)
                        libretro 核心会传入实际采样率

        Returns:
            初始化成功返回 True
        """
        if self._initialized:
            return True

        if sample_rate > 0:
            self._sample_rate = sample_rate

        if self._sample_rate <= 0:
            self._sample_rate = 44100

        try:
            # 初始化 pygame.mixer（如果还没初始化）
            if not pygame.mixer.get_init():
                pygame.mixer.init(
                    frequency=self._sample_rate,
                    size=-16,  # 16-bit signed
                    channels=self.channels,
                    buffer=self.buffer_size
                )

            # 获取一个音频通道用于播放
            # 设置较多的通道数，确保有足够的通道可用
            pygame.mixer.set_num_channels(8)
            self._channel = pygame.mixer.Channel(0)

            self._initialized = True

            actual_freq, actual_size, actual_channels = pygame.mixer.get_init()
            print("Pygame Mixer initialized (pass-through):")
            print(f"  Requested: {self._sample_rate}Hz, {self.channels} ch, buffer={self.buffer_size}")
            print(f"  Actual:    {actual_freq}Hz, {actual_channels} ch, {actual_size}-bit")

            # 更新实际的声道数和采样率
            if actual_freq != self._sample_rate:
                print(f"  Warning: Sample rate mismatch! Using {actual_freq}Hz instead of {self._sample_rate}Hz")
                self._sample_rate = actual_freq

            if actual_channels != self.channels:
                print(f"  Warning: Channel count mismatch! Using {actual_channels} channels instead of {self.channels}")
                self.channels = actual_channels

            return True

        except Exception as e:
            print(f"Failed to initialize Pygame Mixer: {e}")
            import traceback
            traceback.print_exc()
            return False

    def play_samples(self, samples: np.ndarray) -> None:
        """
        播放音频采样

        Args:
            samples: 音频采样数据，numpy 数组格式
                    - int16 格式：直接使用（来自 libretro）
                    - float32 格式：范围 [-1.0, 1.0]（需要转换）

        注意：
            - Ardens 输出 int16 格式，范围 [-SOUND_GAIN, +SOUND_GAIN] (SOUND_GAIN=2000)
            - 我们需要正确处理并避免削波
        """
        if not self._initialized or self._channel is None:
            return

        if samples is None or len(samples) == 0:
            return

        try:
            # 1. 处理不同的输入格式并归一化
            # 参考 Ardens: SOUND_GAIN=2000, 归一化时除以 32768
            if samples.dtype != np.int16:
                samples_int16 = np.asarray(samples, dtype=np.int16)
            else:
                samples_int16 = samples

            # 3. 确保数组是 C-contiguous（pygame.sndarray 要求）
            if not samples_int16.flags['C_CONTIGUOUS']:
                samples_int16 = np.ascontiguousarray(samples_int16)

            # 4. 处理声道转换
            # 注意: pygame.sndarray.make_sound 对单声道的要求:
            # - 单声道 mixer: 需要 1D 数组 (n,)，不能是 (n, 1)
            # - 立体声 mixer: 需要 2D 数组 (n, 2)
            if samples_int16.ndim == 1:
                mono = samples_int16
                if self.channels == 1:
                    audio_data = mono
                else:
                    audio_data = np.repeat(mono[:, None], self.channels, axis=1)
            else:
                data = samples_int16
                if self.channels == 1:
                    audio_data = data[:, 0]
                else:
                    take = min(data.shape[1], self.channels)
                    audio_data = data[:, :take]
                    if take < self.channels:
                        pad = np.repeat(audio_data[:, :1], self.channels - take, axis=1)
                        audio_data = np.concatenate([audio_data, pad], axis=1)

            # 确保是 C-contiguous
            if not audio_data.flags['C_CONTIGUOUS']:
                audio_data = np.ascontiguousarray(audio_data)

            # 5. 创建 Sound 对象并播放
            # pygame.sndarray.make_sound 需要:
            # - 单声道: (n,) 1D 数组
            # - 立体声: (n, 2) 2D 数组
            sound = pygame.sndarray.make_sound(audio_data)

            # 6. 优化的队列管理（保证连续性和低延迟）
            # - 如果通道空闲，直接播放
            # - 如果通道正在播放，始终排队 (保证音频连续,避免断音)
            # - pygame.mixer 会自动管理队列,最多只能排队1个 Sound 对象
            if not self._channel.get_busy():
                self._channel.play(sound)
            else:
                self._channel.queue(sound)

        except Exception as e:
            # 只在第一次错误时打印
            if not hasattr(self, '_error_printed'):
                print(f"Error playing audio: {e}")
                import traceback
                traceback.print_exc()
                self._error_printed = True

    def close(self) -> None:
        """关闭音频驱动"""
        self._initialized = False

        if self._channel:
            try:
                self._channel.stop()
            except:
                pass
            self._channel = None

        # 注意：不要调用 pygame.mixer.quit()，因为可能还有其他组件在使用

class PygameMixerDriverLowLatency(PygameMixerDriver):
    """
    低延迟 Pygame Mixer 音频驱动

    使用更小的缓冲区以降低延迟
    适合对音频同步要求高的场景

    注意：
        - 更小的缓冲区可能增加 CPU 负载
        - 如果系统性能不足，可能出现音频卡顿
    """

    def __init__(self, **kwargs):
        kwargs['buffer_size'] = kwargs.get('buffer_size', 512)  # 更小的缓冲区
        super().__init__(**kwargs)

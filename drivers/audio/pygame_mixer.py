"""
Pygame Mixer 音频驱动

使用 pygame.mixer 播放音频
适合桌面环境（macOS、Windows、Linux）
非阻塞设计，不影响游戏 FPS
"""
import numpy as np
from .base import AudioDriver

try:
    import pygame.mixer
    import pygame.sndarray
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False


class PygameMixerDriver(AudioDriver):
    """
    Pygame Mixer 音频驱动

    使用 pygame.mixer 播放音频流
    非阻塞设计，适合实时游戏音频

    Args:
        sample_rate: 采样率，默认 50000 Hz (Ardens 标准: 16MHz / 320 = 50kHz)
        channels: 声道数，默认 1 (单声道，匹配 Arduboy 硬件)
        buffer_size: 缓冲区大小，默认 2048 (降低延迟)
        volume: 音量，0.0-1.0，默认 0.3

    注意：
        - Ardens 使用 50kHz 采样率 (16,000,000 Hz / 320 cycles = 50,000 Hz)
        - Arduboy 硬件是单声道输出
        - 较小的缓冲区可以降低音频延迟，但需要更频繁的音频更新
    """

    def __init__(
        self,
        sample_rate: int = 48000,  # 匹配 Ardens: 16MHz / 320 = 50kHz
        channels: int = 2,          # 立体声（更好的兼容性）
        buffer_size: int = 1024,    # 降低延迟 (从 4096 降低到 1024)
        volume: float = 0.3
    ):
        super().__init__()

        if not PYGAME_AVAILABLE:
            raise ImportError("pygame is not installed. Please run: pip install pygame")

        self._sample_rate = sample_rate
        self.channels = channels
        self.buffer_size = buffer_size
        self.volume = max(0.0, min(1.0, volume))

        self._initialized = False
        self._channel = None
        self._resampler_state = 0.0  # 用于重采样的状态

        # 音频统计信息（用于调试）
        self._frame_count = 0
        self._total_samples = 0
        self._underrun_count = 0  # 音频队列空的次数

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

        # 更新采样率（从 libretro 核心获取，应该是 50kHz）
        if sample_rate > 0:
            self._sample_rate = sample_rate

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
            self._channel.set_volume(self.volume)

            self._initialized = True

            actual_freq, actual_size, actual_channels = pygame.mixer.get_init()
            print(f"Pygame Mixer initialized:")
            print(f"  Requested: {self._sample_rate}Hz, {self.channels} ch, buffer={self.buffer_size}")
            print(f"  Actual:    {actual_freq}Hz, {actual_channels} ch, {actual_size}-bit")
            print(f"  Volume:    {self.volume}")

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
            if samples.dtype == np.int16:
                # int16 → float → 应用音量 → int16
                # 这样可以避免 int16 直接乘法导致的精度损失
                samples_float = samples.astype(np.float32) / 32768.0  # 归一化到 [-1, 1]
                if self.volume != 1.0:
                    samples_float *= self.volume
                # 防止削波
                samples_float = np.clip(samples_float, -1.0, 1.0)
                samples_int16 = (samples_float * 32767).astype(np.int16)
            else:
                # float32 格式，转换为 int16
                samples_float = samples.astype(np.float32)
                if self.volume != 1.0:
                    samples_float *= self.volume
                # Clip 到 [-1.0, 1.0] 范围（防止爆音）
                samples_float = np.clip(samples_float, -1.0, 1.0)
                # 转换为 int16
                samples_int16 = (samples_float * 32767).astype(np.int16)

            # 3. 确保数组是 C-contiguous（pygame.sndarray 要求）
            if not samples_int16.flags['C_CONTIGUOUS']:
                samples_int16 = np.ascontiguousarray(samples_int16)

            # 4. 处理声道转换
            # 注意: pygame.sndarray.make_sound 对单声道的要求:
            # - 单声道 mixer: 需要 1D 数组 (n,)，不能是 (n, 1)
            # - 立体声 mixer: 需要 2D 数组 (n, 2)
            if samples_int16.ndim == 1:
                if self.channels == 1:
                    # 单声道输出，保持 1D 数组
                    audio_data = samples_int16
                else:
                    # 立体声输出，复制单声道到两个通道
                    audio_data = np.column_stack([samples_int16, samples_int16])
            else:
                # 已经是 2D 数组
                if self.channels == 1:
                    # 如果 mixer 是单声道但数据是 2D，需要转为 1D
                    audio_data = samples_int16.flatten()
                else:
                    audio_data = samples_int16

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
                # 通道空闲，直接播放
                self._channel.play(sound)
                self._underrun_count += 1  # 记录音频中断次数
            else:
                # 通道忙，排队下一个 (pygame 只允许队列1个,会自动替换旧的)
                self._channel.queue(sound)

            # 更新统计信息
            self._frame_count += 1
            self._total_samples += len(samples)

            # 每 300 帧打印一次统计信息（约 5 秒）
            # if self._frame_count % 300 == 0:
            #     avg_samples = self._total_samples / self._frame_count if self._frame_count > 0 else 0
            #     underrun_rate = self._underrun_count / self._frame_count * 100 if self._frame_count > 0 else 0
            #     print(f"[Audio Stats] Frames: {self._frame_count}, "
            #           f"Avg samples/frame: {avg_samples:.1f}, "
            #           f"Underruns: {self._underrun_count} ({underrun_rate:.1f}%)")

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

    def set_volume(self, volume: float) -> None:
        """
        设置音量

        Args:
            volume: 音量，0.0-1.0
        """
        self.volume = max(0.0, min(1.0, volume))
        if self._channel:
            self._channel.set_volume(self.volume)


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

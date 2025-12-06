"""
ALSA 音频驱动
使用 pyalsaaudio 库播放音频
适合在树莓派等 Linux 系统上使用
"""
import numpy as np
from typing import Optional
from .base import AudioDriver

try:
    import alsaaudio
    ALSA_AVAILABLE = True
except ImportError:
    ALSA_AVAILABLE = False


class AlsaAudioDriver(AudioDriver):
    """
    ALSA 音频驱动

    使用 pyalsaaudio 库播放音频，适合树莓派等 Linux 系统

    Args:
        device: ALSA 设备名称，默认 'default'
        sample_rate: 采样率，默认 50000 Hz (Ardens 标准: 16MHz / 320 = 50kHz)
        channels: 声道数，默认 1 (单声道，匹配 Arduboy 硬件)
        buffer_size: 缓冲区大小，默认 2048 (降低延迟)
        volume: 音量，0.0-1.0，默认 0.3

    注意：
        - Ardens 使用 50kHz 采样率 (16,000,000 Hz / 320 cycles = 50,000 Hz)
        - Arduboy 硬件是单声道输出
    """

    def __init__(
        self,
        device: str = 'default',
        sample_rate: int = 44100,  # 匹配 Ardens: 16MHz / 320
        channels: int = 2,          # 单声道（Arduboy 硬件特性）
        buffer_size: int = 4096,    # 降低延迟
        volume: float = 0.3
    ):
        """
        初始化 ALSA 音频驱动

        Args:
            device: ALSA 设备名称
            sample_rate: 采样率
            channels: 声道数（1=单声道，2=立体声）
            buffer_size: 缓冲区大小
            volume: 音量（0.0-1.0）
        """
        if not ALSA_AVAILABLE:
            raise ImportError(
                "alsaaudio library not found. Please install it:\n"
                "  sudo apt-get install python3-pyaudio python3-alsaaudio\n"
                "or\n"
                "  pip3 install pyalsaaudio"
            )

        self.device_name = device
        self._sample_rate = sample_rate  # 使用 _sample_rate（基类定义为 property）
        self.channels = channels
        self.buffer_size = buffer_size
        self.volume = max(0.0, min(1.0, volume))
        self.pcm: Optional[alsaaudio.PCM] = None
        self._running = False

    def init(self, sample_rate: int = 50000) -> bool:
        """
        初始化 ALSA 音频设备

        Args:
            sample_rate: 采样率，默认 50000 Hz (Ardens 标准)
                        libretro 核心会传入实际采样率

        Returns:
            初始化成功返回 True，失败返回 False
        """
        if self._running:
            return True

        self._sample_rate = sample_rate  # 使用 _sample_rate（基类定义为 property）

        try:
            # 创建 PCM 对象（使用非阻塞模式）
            self.pcm = alsaaudio.PCM(
                type=alsaaudio.PCM_PLAYBACK,
                mode=alsaaudio.PCM_NONBLOCK,  # 非阻塞模式，避免阻塞主循环
                device=self.device_name
            )

            # 配置音频参数
            self.pcm.setchannels(self.channels)
            self.pcm.setrate(self._sample_rate)
            self.pcm.setformat(alsaaudio.PCM_FORMAT_S16_LE)  # 16-bit signed little-endian
            self.pcm.setperiodsize(self.buffer_size)

            self._running = True

            print(f"ALSA Audio initialized:")
            print(f"  Device: {self.device_name}")
            print(f"  Sample rate: {self._sample_rate} Hz")
            print(f"  Channels: {self.channels}")
            print(f"  Format: 16-bit signed LE")
            print(f"  Buffer size: {self.buffer_size}")
            print(f"  Volume: {self.volume}")

            return True

        except alsaaudio.ALSAAudioError as e:
            print(f"Failed to initialize ALSA audio: {e}")
            print("\nTroubleshooting:")
            print("  1. Check if ALSA is properly configured: aplay -l")
            print("  2. Test audio: speaker-test -t wav -c 2")
            print("  3. Adjust volume: alsamixer")
            return False
        except Exception as e:
            print(f"Unexpected error initializing ALSA audio: {e}")
            import traceback
            traceback.print_exc()
            return False

    def play_samples(self, samples: np.ndarray) -> None:
        """
        播放音频采样

        Args:
            samples: 音频采样数据，numpy 数组格式
                    - int16 格式：来自 libretro 核心，范围 [-32768, 32767]
                    - float32 格式：归一化范围 [-1.0, 1.0]

        注意：
            - Ardens 输出 int16 格式，范围 [-SOUND_GAIN, +SOUND_GAIN] (SOUND_GAIN=2000)
            - 我们需要正确处理并避免削波
        """
        if not self._running or self.pcm is None:
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

            # 2. 处理声道转换
            if samples_int16.ndim == 1 and self.channels == 2:
                # 单声道转立体声：使用 repeat 优化内存访问
                samples_int16 = np.repeat(samples_int16, 2)

            # 3. 直接写入 ALSA（非阻塞模式，不会阻塞主循环）
            self.pcm.write(samples_int16.tobytes())

        except Exception as e:
            # 只在第一次错误时打印
            if not hasattr(self, '_error_printed'):
                print(f"[ALSA] Error playing samples: {e}")
                import traceback
                traceback.print_exc()
                self._error_printed = True

    def close(self) -> None:
        """关闭 ALSA 音频设备"""
        self._running = False

        if self.pcm:
            try:
                self.pcm.close()
            except Exception:
                pass
            self.pcm = None

        print("ALSA Audio closed")

    def set_volume(self, volume: float) -> None:
        """
        设置音量

        Args:
            volume: 音量，0.0-1.0
        """
        self.volume = max(0.0, min(1.0, volume))

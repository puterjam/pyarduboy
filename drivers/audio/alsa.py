"""
ALSA 音频驱动（直通模式）
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
    """使用 ALSA 直接播放 libretro 输出的 int16 音频，不做额外处理。"""

    def __init__(
        self,
        device: str = 'default',
        sample_rate: Optional[int] = None,
        channels: int = 2,
        buffer_size: int = 256
    ):
        if not ALSA_AVAILABLE:
            raise ImportError(
                "alsaaudio library not found. Please install it:\n"
                "  sudo apt-get install python3-pyaudio python3-alsaaudio\n"
                "or\n"
                "  pip3 install pyalsaaudio"
            )

        self.device_name = device
        self._sample_rate = sample_rate or 0
        self.channels = channels
        self.buffer_size = buffer_size
        self.pcm: Optional[alsaaudio.PCM] = None
        self._running = False
        self._write_errors = 0

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

        if sample_rate > 0:
            self._sample_rate = sample_rate

        if self._sample_rate <= 0:
            self._sample_rate = 44100

        try:
            # 创建 PCM 对象（使用非阻塞模式）
            self.pcm = alsaaudio.PCM(
                type=alsaaudio.PCM_PLAYBACK,
                mode=alsaaudio.PCM_NONBLOCK,  # 非阻塞模式，避免阻塞主循环
                device=self.device_name
            )

            # 选择音频格式
            audio_format = alsaaudio.PCM_FORMAT_S16_LE
            bit_depth = 16

            # 配置音频参数
            self.pcm.setchannels(self.channels)
            self.pcm.setrate(self._sample_rate)
            self.pcm.setformat(audio_format)  # 24-bit 或 16-bit signed little-endian
            self.pcm.setperiodsize(self.buffer_size)

            self._running = True

            print(f"ALSA Audio initialized:")
            print(f"  Device: {self.device_name}")
            print(f"  Sample rate: {self._sample_rate} Hz")
            print(f"  Channels: {self.channels}")
            print(f"  Format: {bit_depth}-bit signed LE")
            print(f"  Buffer size: {self.buffer_size}")

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
            if samples.dtype != np.int16:
                samples_int16 = np.asarray(samples, dtype=np.int16)
            else:
                samples_int16 = samples

            if not samples_int16.flags['C_CONTIGUOUS']:
                samples_int16 = np.ascontiguousarray(samples_int16)

            audio_data = samples_int16.tobytes()

            try:
                self.pcm.write(audio_data)
            except alsaaudio.ALSAAudioError:
                # 非阻塞模式下，缓冲区满时会抛出异常，这是正常的
                self._write_errors += 1
                pass

            # 更新统计信息
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

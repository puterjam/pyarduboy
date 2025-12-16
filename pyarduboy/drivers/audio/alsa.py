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

        # 音频增益自动校正（解决多线程环境下增益不一致的问题）
        self._auto_gain = True
        self._target_peak = 2000  # Ardens 标准音量峰值
        self._detected_peak = None

        # 调试开关
        self._debug = False  # 设置为 True 启用详细调试信息

    def init(self, sample_rate: int = 44100) -> bool:
        """
        初始化 ALSA 音频设备

        Args:
            sample_rate: 采样率，默认 44100 Hz (Ardens 标准)
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

        注意：
            - Ardens 输出 int16 格式，范围 [-SOUND_GAIN, +SOUND_GAIN] (SOUND_GAIN=2000)
            - 我们需要正确处理并避免削波
        """
        if not self._running or self.pcm is None:
            return

        if samples is None or len(samples) == 0:
            return

        try:
            # 调试信息：监控音频数据
            if self._debug:
                if not hasattr(self, '_frame_count'):
                    self._frame_count = 0
                    self._non_zero_count = 0

                self._frame_count += 1
                max_val = abs(samples).max()

                # 检测到非零音频时打印
                if max_val > 0:
                    self._non_zero_count += 1
                    if self._non_zero_count <= 5:  # 只打印前5次非零音频
                        print(f"[ALSA Debug] Frame {self._frame_count} - Non-zero audio detected!")
                        print(f"  Shape: {samples.shape}")
                        print(f"  Dtype: {samples.dtype}")
                        print(f"  Min/Max: {samples.min()}/{samples.max()}")
                        print(f"  Mean: {samples.mean():.2f}")
                        print(f"  First 20 samples: {samples.flatten()[:20]}")

                # 每1000帧统计一次
                if self._frame_count % 1000 == 0:
                    print(f"[ALSA Stats] {self._frame_count} frames, {self._non_zero_count} non-zero ({100*self._non_zero_count/self._frame_count:.1f}%)")
            else:
                # 即使不调试，也需要计算 max_val 用于增益控制
                max_val = abs(samples).max()

            # 转换为 int16
            if samples.dtype != np.int16:
                samples_int16 = np.asarray(samples, dtype=np.int16)
            else:
                samples_int16 = samples

            # 自动增益校正
            if self._auto_gain and max_val > 0:
                # 检测峰值（使用前几帧非零音频的最大值）
                if self._detected_peak is None:
                    # 初始化非零计数器（如果未开启调试模式）
                    if not hasattr(self, '_non_zero_count'):
                        self._non_zero_count = 0
                    self._non_zero_count += 1

                    if self._non_zero_count <= 10:
                        current_peak = abs(samples_int16).max()
                        if current_peak > self._target_peak * 1.5:  # 峰值明显高于标准值
                            self._detected_peak = current_peak
                            print(f"[ALSA] Auto-gain detected peak: {current_peak}, target: {self._target_peak}")
                            print(f"[ALSA] Gain correction factor: {self._target_peak / current_peak:.3f}")

                # 应用增益校正
                if self._detected_peak and self._detected_peak > self._target_peak:
                    gain_factor = self._target_peak / self._detected_peak
                    samples_int16 = (samples_int16.astype(np.float32) * gain_factor).astype(np.int16)

            if not samples_int16.flags['C_CONTIGUOUS']:
                samples_int16 = np.ascontiguousarray(samples_int16)

            audio_data = samples_int16.tobytes()

            try:
                self.pcm.write(audio_data)
            except alsaaudio.ALSAAudioError as e:
                # 非阻塞模式下，缓冲区满时会抛出异常，这是正常的
                self._write_errors += 1
                if self._write_errors % 100 == 1:  # 每100次丢帧警告一次
                    print(f"[ALSA] Buffer full, dropped {self._write_errors} frames: {e}")

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

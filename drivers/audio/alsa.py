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
        sample_rate: 采样率，默认 48000 Hz (专业音频标准)
        channels: 声道数，默认 2 (立体声，更好的兼容性)
        buffer_size: 缓冲区大小，默认 1024 (降低延迟)
        volume: 音量，0.0-1.0，默认 0.3
        use_24bit: 使用 24-bit 音频，默认 True (更高音质)

    注意：
        - 使用 48kHz 采样率以匹配系统默认音频设备
        - 立体声输出提供更好的兼容性
        - 24-bit 音频提供更高的动态范围和音质
    """

    def __init__(
        self,
        device: str = 'default',
        sample_rate: int = 48000,  # 专业音频标准采样率
        channels: int = 2,          # 立体声（更好的兼容性）
        buffer_size: int = 1024,    # 降低延迟 (从 4096 降低到 1024)
        volume: float = 0.3,
        use_24bit: bool = True      # 使用 24-bit 音频（更高音质）
    ):
        """
        初始化 ALSA 音频驱动

        Args:
            device: ALSA 设备名称
            sample_rate: 采样率
            channels: 声道数（1=单声道，2=立体声）
            buffer_size: 缓冲区大小
            volume: 音量（0.0-1.0）
            use_24bit: 使用 24-bit 音频（更高音质）
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
        self.use_24bit = use_24bit
        self.pcm: Optional[alsaaudio.PCM] = None
        self._running = False

        # 音频统计信息（用于调试）
        self._frame_count = 0
        self._total_samples = 0
        self._write_errors = 0  # 写入错误次数

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

            # 选择音频格式
            audio_format = alsaaudio.PCM_FORMAT_S24_3LE if self.use_24bit else alsaaudio.PCM_FORMAT_S16_LE
            bit_depth = 24 if self.use_24bit else 16

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
            # 1. 处理不同的输入格式并归一化到 float32
            # 参考 Ardens: SOUND_GAIN=2000, 归一化时除以 32768
            if samples.dtype == np.int16:
                # int16 → float32 (归一化到 [-1.0, 1.0])
                samples_float = samples.astype(np.float32) / 32768.0
            else:
                # 已经是 float32 格式
                samples_float = samples.astype(np.float32)

            # 2. 应用音量控制
            if self.volume != 1.0:
                samples_float *= self.volume

            # 防止削波
            samples_float = np.clip(samples_float, -1.0, 1.0)

            # 3. 处理声道转换（在浮点域）
            if samples_float.ndim == 1 and self.channels == 2:
                # 单声道转立体声：使用 repeat 优化内存访问
                samples_float = np.repeat(samples_float, 2)

            # 4. 转换为目标位深度格式
            if self.use_24bit:
                # 转换为 24-bit (3 bytes per sample, little-endian)
                # int24 范围: -8388608 to 8388607 (2^23)
                samples_int32 = (samples_float * 8388607.0).astype(np.int32)

                # 手动打包为 3-byte little-endian 格式
                num_samples = len(samples_int32)
                audio_data = bytearray(num_samples * 3)

                for i, sample in enumerate(samples_int32):
                    # 将 int32 转换为 3-byte little-endian
                    # 处理负数：使用补码表示
                    if sample < 0:
                        sample = sample & 0xFFFFFF  # 24-bit 补码
                    audio_data[i*3] = sample & 0xFF
                    audio_data[i*3 + 1] = (sample >> 8) & 0xFF
                    audio_data[i*3 + 2] = (sample >> 16) & 0xFF

                audio_data = bytes(audio_data)
            else:
                # 转换为 16-bit
                samples_int16 = (samples_float * 32767).astype(np.int16)
                audio_data = samples_int16.tobytes()

            # 5. 直接写入 ALSA（非阻塞模式，不会阻塞主循环）
            try:
                self.pcm.write(audio_data)
            except alsaaudio.ALSAAudioError:
                # 非阻塞模式下，缓冲区满时会抛出异常，这是正常的
                self._write_errors += 1
                pass

            # 更新统计信息
            self._frame_count += 1
            self._total_samples += len(samples)

            # 每 300 帧打印一次统计信息（约 5 秒 @ 60 FPS）
            # if self._frame_count % 300 == 0:
            #     avg_samples = self._total_samples / self._frame_count if self._frame_count > 0 else 0
            #     error_rate = self._write_errors / self._frame_count * 100 if self._frame_count > 0 else 0
            #     print(f"[ALSA Stats] Frames: {self._frame_count}, "
            #           f"Avg samples/frame: {avg_samples:.1f}, "
            #           f"Write errors: {self._write_errors} ({error_rate:.1f}%)")

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

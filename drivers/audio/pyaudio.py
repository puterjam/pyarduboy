"""
PyAudio 音频驱动

使用 PyAudio 库播放音频
适合桌面环境（macOS、Windows、Linux）
基于 PortAudio，跨平台低延迟音频库
使用 callback 模式实现真正的非阻塞
"""
import numpy as np
import threading
from collections import deque
from typing import Optional
from .base import AudioDriver

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False


class PyAudioDriver(AudioDriver):
    """
    PyAudio 音频驱动 (Callback 模式)

    使用 PyAudio callback 模式播放音频流
    真正的非阻塞实现，不会影响主循环 FPS

    Args:
        sample_rate: 采样率，默认 50000 Hz (Ardens 标准: 16MHz / 320 = 50kHz)
        channels: 声道数，默认 1 (单声道，匹配 Arduboy 硬件)
        chunk_size: 每次回调的块大小，默认 2048 帧 (降低延迟)
        volume: 音量，0.0-1.0，默认 0.3

    注意：
        - Ardens 使用 50kHz 采样率 (16,000,000 Hz / 320 cycles = 50,000 Hz)
        - Arduboy 硬件是单声道输出
        - callback 模式实现真正的非阻塞音频播放
    """

    def __init__(
        self,
        sample_rate: int = 50000,  # 匹配 Ardens: 16MHz / 320
        channels: int = 1,          # 单声道（Arduboy 硬件特性）
        buffer_size: int = 2048,    # 降低延迟（别名：chunk_size）
        volume: float = 0.3,
        chunk_size: int = None      # 向后兼容（已弃用，使用 buffer_size）
    ):
        super().__init__()

        if not PYAUDIO_AVAILABLE:
            raise ImportError("PyAudio is not installed. Please run: pip install pyaudio")

        self._sample_rate = sample_rate
        self.channels = channels
        # 统一使用 buffer_size（兼容旧的 chunk_size 参数）
        self.buffer_size = chunk_size if chunk_size is not None else buffer_size
        self.volume = max(0.0, min(1.0, volume))

        self._pyaudio: Optional[pyaudio.PyAudio] = None
        self._stream: Optional[pyaudio.Stream] = None
        self._running = False

        # 音频缓冲队列（线程安全）
        self._audio_buffer = deque()
        self._buffer_lock = threading.Lock()

    def _audio_callback(self, _in_data, frame_count, _time_info, _status):
        """
        PyAudio callback 函数（在独立线程中调用）

        Args:
            _in_data: 输入数据（未使用）
            frame_count: 需要的帧数
            _time_info: 时间信息（未使用）
            _status: 状态标志（未使用）

        Returns:
            (audio_data, paContinue)
        """
        needed_bytes = frame_count * self.channels * 2  # int16 = 2 bytes

        with self._buffer_lock:
            if self._audio_buffer:
                # 从队列中取出数据
                audio_data = bytearray()

                # 尽可能多地从队列中取数据，直到满足 needed_bytes 或队列为空
                while len(audio_data) < needed_bytes and self._audio_buffer:
                    chunk = self._audio_buffer.popleft()
                    audio_data.extend(chunk)

                # 如果数据足够，截取需要的部分
                if len(audio_data) >= needed_bytes:
                    result = bytes(audio_data[:needed_bytes])
                    # 如果有多余的数据，放回队列头部
                    remaining = audio_data[needed_bytes:]
                    if remaining:
                        self._audio_buffer.appendleft(bytes(remaining))
                else:
                    # 数据不足，补充静音
                    result = bytes(audio_data) + bytes(needed_bytes - len(audio_data))
            else:
                # 队列为空，播放静音
                result = bytes(needed_bytes)

        return (result, pyaudio.paContinue)

    def init(self, sample_rate: int = 50000) -> bool:
        """
        初始化音频驱动

        Args:
            sample_rate: 采样率，默认 50000 Hz (Ardens 标准)
                        libretro 核心会传入实际采样率

        Returns:
            初始化成功返回 True
        """
        if self._running:
            return True

        # 更新采样率（从 libretro 核心获取，应该是 50kHz）
        if sample_rate > 0:
            self._sample_rate = sample_rate

        try:
            # 创建 PyAudio 实例
            self._pyaudio = pyaudio.PyAudio()

            # 打开音频流（callback 模式 - 真正的非阻塞）
            self._stream = self._pyaudio.open(
                format=pyaudio.paInt16,  # 16-bit signed integer
                channels=self.channels,
                rate=self._sample_rate,
                output=True,
                frames_per_buffer=self.buffer_size,
                stream_callback=self._audio_callback  # 使用 callback 模式
            )

            # 启动音频流
            self._stream.start_stream()
            self._running = True

            print(f"PyAudio initialized (callback mode - non-blocking):")
            print(f"  Sample rate: {self._sample_rate} Hz")
            print(f"  Channels: {self.channels}")
            print(f"  Format: 16-bit signed")
            print(f"  Buffer size: {self.buffer_size}")
            print(f"  Volume: {self.volume}")
            return True

        except Exception as e:
            print(f"Failed to initialize PyAudio: {e}")
            import traceback
            traceback.print_exc()
            if self._stream:
                self._stream.close()
            if self._pyaudio:
                self._pyaudio.terminate()
            return False

    def play_samples(self, samples: np.ndarray) -> None:
        """
        播放音频采样（添加到缓冲队列）

        Args:
            samples: 音频采样数据，numpy 数组格式
                    - int16 格式：直接使用（来自 libretro）
                    - float32 格式：范围 [-1.0, 1.0]（需要转换）

        注意：
            - Ardens 输出 int16 格式，范围 [-SOUND_GAIN, +SOUND_GAIN] (SOUND_GAIN=2000)
            - 我们需要正确处理并避免削波
        """
        if not self._running or self._stream is None:
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

            # 3. 处理声道转换
            if samples_int16.ndim == 1 and self.channels == 2:
                # 单声道转立体声：使用 repeat 优化内存访问
                samples_int16 = np.repeat(samples_int16, 2)

            # 4. 添加到缓冲队列（线程安全，非阻塞）
            audio_data = samples_int16.tobytes()

            with self._buffer_lock:
                # 限制缓冲队列大小，避免延迟累积
                # 最多保留约 20 帧的数据（约 333ms @ 60 FPS）
                max_buffer_items = 20
                if len(self._audio_buffer) < max_buffer_items:
                    self._audio_buffer.append(audio_data)
                else:
                    # 队列满了，移除最旧的数据（避免延迟累积）
                    self._audio_buffer.popleft()
                    self._audio_buffer.append(audio_data)

        except Exception as e:
            # 只在第一次错误时打印
            if not hasattr(self, '_error_printed'):
                print(f"[PyAudio] Error playing samples: {e}")
                import traceback
                traceback.print_exc()
                self._error_printed = True

    def close(self) -> None:
        """关闭音频驱动"""
        self._running = False

        if self._stream:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

        if self._pyaudio:
            try:
                self._pyaudio.terminate()
            except Exception:
                pass
            self._pyaudio = None

        # 清空缓冲队列
        with self._buffer_lock:
            self._audio_buffer.clear()

        print("PyAudio closed")

    def set_volume(self, volume: float) -> None:
        """
        设置音量

        Args:
            volume: 音量，0.0-1.0
        """
        self.volume = max(0.0, min(1.0, volume))


class PyAudioDriverLowLatency(PyAudioDriver):
    """
    低延迟 PyAudio 音频驱动

    使用更小的缓冲区以降低延迟
    适合对音频同步要求高的场景

    注意：
        - 更小的缓冲区可能增加 CPU 负载
        - 如果系统性能不足，可能出现音频卡顿
    """

    def __init__(self, **kwargs):
        kwargs['buffer_size'] = kwargs.get('buffer_size', 512)
        super().__init__(**kwargs)


class PyAudioDriverHighQuality(PyAudioDriver):
    """
    高质量 PyAudio 音频驱动

    使用更大的缓冲区以提高音质和稳定性
    适合对稳定性要求高的场景
    """

    def __init__(self, **kwargs):
        kwargs['buffer_size'] = kwargs.get('buffer_size', 4096)
        super().__init__(**kwargs)

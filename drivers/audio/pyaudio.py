"""
PyAudio 音频驱动

使用 PyAudio 库播放音频（完全直通，不做额外处理）
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

    使用 PyAudio callback 模式播放 libretro 输出的 int16 音频流，
    不做重采样、混音或音量处理，确保与核心输出完全一致。
    """

    def __init__(
        self,
        sample_rate: Optional[int] = None,
        channels: int = 2,
        buffer_size: int = 256,
        use_24bit: bool = False
    ):
        super().__init__()

        if not PYAUDIO_AVAILABLE:
            raise ImportError("PyAudio is not installed. Please run: pip install pyaudio")

        self._sample_rate = sample_rate or 0
        self.channels = channels
        self.buffer_size = buffer_size
        self.use_24bit = use_24bit

        self._pyaudio: Optional[pyaudio.PyAudio] = None
        self._stream: Optional[pyaudio.Stream] = None
        self._running = False

        # 音频缓冲队列（线程安全）
        self._audio_buffer = deque()
        self._buffer_lock = threading.Lock()

        # 音频统计信息（用于调试）
        self._frame_count = 0
        self._total_samples = 0
        self._underrun_count = 0  # 缓冲区空的次数

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
        # 计算需要的字节数：24-bit = 3 bytes, 16-bit = 2 bytes
        bytes_per_sample = 3 if self.use_24bit else 2
        needed_bytes = frame_count * self.channels * bytes_per_sample

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
                    self._underrun_count += 1  # 记录缓冲区不足次数
            else:
                # 队列为空，播放静音
                result = bytes(needed_bytes)
                self._underrun_count += 1  # 记录缓冲区空次数

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

        if sample_rate > 0:
            self._sample_rate = sample_rate

        if self._sample_rate <= 0:
            self._sample_rate = 44100  # 回退到常见桌面采样率

        try:
            # 创建 PyAudio 实例
            self._pyaudio = pyaudio.PyAudio()

            # 选择音频格式
            audio_format = pyaudio.paInt24 if self.use_24bit else pyaudio.paInt16
            bit_depth = 24 if self.use_24bit else 16

            # 打开音频流（callback 模式 - 真正的非阻塞）
            self._stream = self._pyaudio.open(
                format=audio_format,
                channels=self.channels,
                rate=self._sample_rate,
                output=True,
                frames_per_buffer=self.buffer_size,
                stream_callback=self._audio_callback  # 使用 callback 模式
            )

            # 启动音频流
            self._stream.start_stream()
            self._running = True

            print("PyAudio initialized (callback mode - pass-through):")
            print(f"  Sample rate: {self._sample_rate} Hz")
            print(f"  Channels: {self.channels}")
            print(f"  Format: {bit_depth}-bit signed")
            print(f"  Buffer size: {self.buffer_size}")
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
            if samples.dtype != np.int16:
                samples_int16 = np.asarray(samples, dtype=np.int16)
            else:
                samples_int16 = samples

            if not samples_int16.flags['C_CONTIGUOUS']:
                samples_int16 = np.ascontiguousarray(samples_int16)

            audio_data = samples_int16.tobytes()

            with self._buffer_lock:
                # 限制缓冲队列大小，避免延迟累积
                # 保持约 15 帧的数据（约 250ms @ 60 FPS）提供足够缓冲
                max_buffer_items = 15
                if len(self._audio_buffer) < max_buffer_items:
                    self._audio_buffer.append(audio_data)
                else:
                    # 队列满了，移除最旧的数据（避免延迟累积）
                    self._audio_buffer.popleft()
                    self._audio_buffer.append(audio_data)

            # 更新统计信息
            self._frame_count += 1
            self._total_samples += len(samples)

            # 每 300 帧打印一次统计信息（约 5 秒 @ 60 FPS）
            # if self._frame_count % 300 == 0:
            #     avg_samples = self._total_samples / self._frame_count if self._frame_count > 0 else 0
            #     underrun_rate = self._underrun_count / self._frame_count * 100 if self._frame_count > 0 else 0
            #     buffer_len = len(self._audio_buffer)
            #     print(f"[PyAudio] Frames: {self._frame_count}, "
            #           f"Avg: {avg_samples:.1f} samples/frame, "
            #           f"Underruns: {self._underrun_count} ({underrun_rate:.1f}%), "
            #           f"Queue: {buffer_len}")

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

        with self._buffer_lock:
            self._audio_buffer.clear()

        print("PyAudio closed")


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

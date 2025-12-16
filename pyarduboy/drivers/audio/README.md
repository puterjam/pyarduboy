# 音频驱动开发指南

## 概述

本目录包含 PyArduboy 项目的音频驱动实现，用于在不同平台上播放 Arduboy 游戏的音频。

## Arduboy 音频规格

### 硬件特性
- **采样率**: 50000 Hz (50kHz)
  - 计算公式: 16 MHz CPU / 320 cycles = 50,000 Hz
  - 这是 Ardens 模拟器核心使用的标准采样率
- **声道**: 单声道 (Mono)
  - Arduboy 硬件使用单个蜂鸣器输出音频
- **音频格式**: 16-bit signed integer (int16)
  - 数据范围: [-32768, 32767]
  - Ardens 使用 SOUND_GAIN=2000 作为振幅

### 音频数据流程
```
Ardens 核心 (libretro)
    ↓ int16 格式，50kHz，单声道
音频驱动 (AudioDriver)
    ↓ 格式转换、音量控制
硬件输出 (扬声器/耳机)
```

## 驱动架构

### 基类接口

所有音频驱动必须继承 `AudioDriver` 基类并实现以下接口：

```python
from pyarduboy.drivers.audio.base import AudioDriver
import numpy as np

class MyAudioDriver(AudioDriver):
    def init(self, sample_rate: int = 50000) -> bool:
        """
        初始化音频驱动

        Args:
            sample_rate: 采样率，默认 50000 Hz (Ardens 标准)
                        libretro 核心会在运行时传入实际采样率

        Returns:
            初始化成功返回 True，失败返回 False
        """
        pass

    def play_samples(self, samples: np.ndarray) -> None:
        """
        播放音频采样

        Args:
            samples: 音频采样数据，numpy 数组格式
                    - int16 格式：来自 libretro 核心，范围 [-32768, 32767]
                    - float32 格式：归一化范围 [-1.0, 1.0]（某些驱动使用）
        """
        pass

    def close(self) -> None:
        """关闭音频驱动，释放资源"""
        pass
```

### 基类属性

基类提供以下属性：

- `self._sample_rate`: 当前采样率 (int)
- `self._running`: 驱动是否正在运行 (bool)
- `self.is_running`: 只读属性，返回运行状态
- `self.sample_rate`: 只读属性，返回采样率

## 现有驱动

### 1. NullAudioDriver (null.py)
**用途**: 无声模式，不播放任何音频

**特点**:
- 最轻量级，无任何依赖
- 用于测试或无需音频的场景
- 直接丢弃所有音频数据

**使用场景**:
- 自动化测试
- 无音频硬件的环境
- 性能测试（排除音频影响）

### 2. AlsaAudioDriver (alsa.py)
**用途**: Linux 系统（特别是树莓派）

**特点**:
- 使用 `pyalsaaudio` 库
- 直接访问 ALSA (Advanced Linux Sound Architecture)
- 非阻塞模式，不会阻塞主循环
- 低延迟，适合嵌入式系统

**依赖安装**:
```bash
sudo apt-get install python3-alsaaudio
# 或
pip install pyalsaaudio
```

**配置参数**:
- `device`: ALSA 设备名称，默认 `'default'`
- `sample_rate`: 采样率，默认 `50000` Hz
- `channels`: 声道数，默认 `1` (单声道)
- `buffer_size`: 缓冲区大小，默认 `2048`
- `volume`: 音量 (0.0-1.0)，默认 `0.3`

**适用平台**:
- ✅ Raspberry Pi OS
- ✅ Ubuntu/Debian Linux
- ✅ 其他 Linux 发行版
- ❌ macOS / Windows

### 3. PyAudioDriver (pyaudio.py)
**用途**: 跨平台桌面环境

**特点**:
- 使用 `pyaudio` 库（基于 PortAudio）
- Callback 模式，真正的非阻塞实现
- 跨平台支持
- 线程安全的音频缓冲队列
- 支持音量控制

**依赖安装**:
```bash
pip install pyaudio
```

**配置参数**:
- `sample_rate`: 采样率，默认 `50000` Hz
- `channels`: 声道数，默认 `1` (单声道)
- `buffer_size`: 缓冲区大小，默认 `2048`
- `volume`: 音量 (0.0-1.0)，默认 `0.3`

**适用平台**:
- ✅ macOS
- ✅ Windows
- ✅ Linux (桌面)
- ⚠️  Raspberry Pi (可用，但 ALSA 更高效)

**变体**:
- `PyAudioDriverLowLatency`: 低延迟版本 (buffer_size=512)
- `PyAudioDriverHighQuality`: 高质量版本 (buffer_size=4096)

### 4. PygameMixerDriver (pygame_mixer.py)
**用途**: 游戏开发环境

**特点**:
- 使用 `pygame.mixer` 播放音频
- 适合已集成 Pygame 的项目
- 智能队列管理，降低延迟
- 支持音量控制

**依赖安装**:
```bash
pip install pygame
```

**配置参数**:
- `sample_rate`: 采样率，默认 `50000` Hz
- `channels`: 声道数，默认 `1` (单声道)
- `buffer_size`: 缓冲区大小，默认 `2048`
- `volume`: 音量 (0.0-1.0)，默认 `0.3`

**适用平台**:
- ✅ macOS
- ✅ Windows
- ✅ Linux

**变体**:
- `PygameMixerDriverLowLatency`: 低延迟版本 (buffer_size=512)

## 开发新驱动

### 1. 创建驱动文件

在 `pyarduboy/drivers/audio/` 目录下创建新文件，例如 `my_driver.py`：

```python
"""
我的音频驱动
"""
import numpy as np
from .base import AudioDriver

class MyAudioDriver(AudioDriver):
    """
    我的音频驱动描述

    Args:
        sample_rate: 采样率，默认 50000 Hz (Ardens 标准)
        其他参数...
    """

    def __init__(self, sample_rate: int = 50000, **kwargs):
        super().__init__()
        self._sample_rate = sample_rate
        # 其他初始化...

    def init(self, sample_rate: int = 50000) -> bool:
        """初始化驱动"""
        if self._running:
            return True

        self._sample_rate = sample_rate

        try:
            # 初始化音频设备...
            self._running = True
            return True
        except Exception as e:
            print(f"Failed to initialize: {e}")
            return False

    def play_samples(self, samples: np.ndarray) -> None:
        """播放音频采样"""
        if not self._running:
            return

        if samples is None or len(samples) == 0:
            return

        try:
            # 处理音频数据...
            pass
        except Exception as e:
            print(f"Error playing samples: {e}")

    def close(self) -> None:
        """关闭驱动"""
        self._running = False
        # 释放资源...
```

### 2. 音频格式转换

大多数驱动需要将输入数据转换为 int16 格式：

```python
def play_samples(self, samples: np.ndarray) -> None:
    # 1. 处理 int16 格式（来自 libretro）
    if samples.dtype == np.int16:
        # 可选：应用音量控制
        samples_float = samples.astype(np.float32) / 32768.0  # 归一化到 [-1, 1]
        samples_float *= volume  # 应用音量
        samples_float = np.clip(samples_float, -1.0, 1.0)  # 防止削波
        samples_int16 = (samples_float * 32767).astype(np.int16)
    else:
        # 2. 处理 float32 格式
        samples_float = samples.astype(np.float32)
        samples_float = np.clip(samples_float, -1.0, 1.0)  # 防止削波
        samples_int16 = (samples_float * 32767).astype(np.int16)

    # 3. 声道转换（如果需要）
    if samples_int16.ndim == 1 and self.channels == 2:
        # 单声道转立体声
        samples_int16 = np.repeat(samples_int16, 2)

    # 4. 播放音频
    self.play_audio(samples_int16)
```

### 3. 非阻塞设计

⚠️ **重要**: 音频驱动必须是非阻塞的，否则会影响游戏主循环的 FPS。

**推荐方案**:

1. **Callback 模式** (最佳)
   - 音频在独立线程中播放
   - 主循环只负责向缓冲队列添加数据
   - 示例: PyAudioDriver

2. **非阻塞 API**
   - 使用音频库的非阻塞模式
   - 避免等待音频播放完成
   - 示例: AlsaAudioDriver (PCM_NONBLOCK)

3. **队列管理**
   - 限制队列大小，避免延迟累积
   - 当队列满时，丢弃旧数据或跳过当前帧

```python
# 错误示例 - 会阻塞主循环
def play_samples(self, samples):
    sound.play()
    sound.wait()  # ❌ 阻塞等待播放完成

# 正确示例 - 非阻塞
def play_samples(self, samples):
    if not self.channel.get_busy():
        self.channel.play(sound)  # ✅ 非阻塞播放
    elif not self.channel.get_queue():
        self.channel.queue(sound)  # ✅ 队列下一个
    # else: 队列已满，跳过避免延迟累积
```

### 4. 错误处理

- 初始化失败时返回 `False`
- 播放错误时打印错误信息但不崩溃
- 提供有用的故障排除信息

```python
def init(self, sample_rate: int = 50000) -> bool:
    try:
        # 初始化代码...
        return True
    except Exception as e:
        print(f"Failed to initialize audio: {e}")
        print("\nTroubleshooting:")
        print("  1. Check dependencies: pip install xxx")
        print("  2. Test audio: xxx command")
        return False
```

### 5. 注册驱动

在 `__init__.py` 中注册新驱动：

```python
from .my_driver import MyAudioDriver

__all__ = [
    "AudioDriver",
    "NullAudioDriver",
    "AlsaAudioDriver",
    "PyAudioDriver",
    "PygameMixerDriver",
    "MyAudioDriver",  # 添加新驱动
]
```

## 接口一致性检查清单

所有驱动必须满足以下要求：

- ✅ 默认采样率: `50000 Hz`
- ✅ 默认声道数: `1` (单声道)
- ✅ `init()` 方法签名: `init(self, sample_rate: int = 50000) -> bool`
- ✅ `play_samples()` 方法签名: `play_samples(self, samples: np.ndarray) -> None`
- ✅ `close()` 方法签名: `close(self) -> None`
- ✅ 非阻塞设计（不影响主循环 FPS）
- ✅ 线程安全（如果使用多线程）
- ✅ 错误处理（初始化失败返回 False）
- ✅ 文档完整（docstring 说明参数和用途）

## 测试驱动

### 单元测试

创建测试文件 `test_audio_drivers.py`:

```python
import numpy as np
from pyarduboy.drivers.audio import MyAudioDriver

def test_init():
    driver = MyAudioDriver()
    assert driver.init(50000) == True
    assert driver.sample_rate == 50000
    assert driver.is_running == True

def test_play_samples():
    driver = MyAudioDriver()
    driver.init()

    # 测试 int16 格式
    samples = np.array([0, 1000, -1000, 2000, -2000], dtype=np.int16)
    driver.play_samples(samples)

    # 测试 float32 格式
    samples = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
    driver.play_samples(samples)

    driver.close()

def test_edge_cases():
    driver = MyAudioDriver()
    driver.init()

    # 空数组
    driver.play_samples(np.array([], dtype=np.int16))

    # None
    driver.play_samples(None)

    driver.close()
```

### 集成测试

使用实际的 Arduboy ROM 测试音频播放：

```python
from pyarduboy import Arduboy
from pyarduboy.drivers.audio import MyAudioDriver

# 创建模拟器实例，使用新驱动
emu = Arduboy(
    rom_path="game.hex",
    audio_driver=MyAudioDriver()
)

# 运行游戏并测试音频
emu.run()
```

### 性能测试

测试驱动是否影响游戏 FPS：

```python
import time

driver = MyAudioDriver()
driver.init()

# 模拟游戏循环
fps_samples = []
for _ in range(600):  # 10 秒 @ 60 FPS
    start = time.time()

    # 生成一帧音频数据 (50000 Hz / 60 FPS ≈ 833 samples)
    samples = np.random.randint(-2000, 2000, 833, dtype=np.int16)
    driver.play_samples(samples)

    # 模拟其他游戏逻辑
    time.sleep(1/60)

    fps_samples.append(1.0 / (time.time() - start))

avg_fps = np.mean(fps_samples)
print(f"Average FPS: {avg_fps:.2f}")
assert avg_fps >= 59.0, "Audio driver is blocking the main loop!"

driver.close()
```

## 平台选择指南

| 平台 | 推荐驱动 | 备选驱动 |
|------|---------|---------|
| Raspberry Pi | `AlsaAudioDriver` | `PyAudioDriver` |
| Linux 桌面 | `PyAudioDriver` | `PygameMixerDriver` |
| macOS | `PyAudioDriver` | `PygameMixerDriver` |
| Windows | `PyAudioDriver` | `PygameMixerDriver` |
| 无音频/测试 | `NullAudioDriver` | - |

## 常见问题

### 音频延迟过高
- 减小 `buffer_size` / `period_size`
- 使用低延迟驱动变体
- 检查系统音频设置

### 音频卡顿/爆音
- 增大 `buffer_size` / `period_size`
- 检查 CPU 负载
- 确保驱动是非阻塞的

### 采样率不匹配
- 确保使用 50000 Hz (Ardens 标准)
- 某些音频设备不支持 50kHz，会自动重采样

### 音量过大/过小
- 调整 `volume` 参数 (0.0-1.0)
- 检查系统音量设置
- 注意: Ardens 使用 SOUND_GAIN=2000，相对较小

## 参考资源

- [Arduboy 官方规格](https://arduboy.com/)
- [Ardens 模拟器源码](https://github.com/tiberiusbrown/Ardens)
- [ALSA 文档](https://www.alsa-project.org/)
- [PortAudio 文档](http://www.portaudio.com/)
- [Pygame 音频文档](https://www.pygame.org/docs/ref/mixer.html)

## 贡献

欢迎贡献新的音频驱动！请确保：

1. 遵循上述接口规范
2. 采样率默认为 50000 Hz
3. 实现非阻塞播放
4. 包含完整的文档和错误处理
5. 通过所有测试

提交 Pull Request 前请更新此 README。

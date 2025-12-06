# 音频驱动接口规范

## 概述

本文档定义了所有音频驱动必须遵循的统一接口规范，确保驱动之间的一致性和可互换性。

## 接口规范版本

**版本**: 1.0
**日期**: 2025-12-04

## 统一参数规范

### 构造函数参数

所有音频驱动的 `__init__()` 方法应遵循以下参数命名和默认值：

| 参数名 | 类型 | 默认值 | 说明 | 必需 |
|--------|------|--------|------|------|
| `sample_rate` | `int` | `50000` | 采样率 (Hz)，匹配 Ardens: 16MHz / 320 | 否 |
| `channels` | `int` | `1` | 声道数（1=单声道，2=立体声），匹配 Arduboy 硬件 | 否 |
| `buffer_size` | `int` | `2048` | 缓冲区大小（帧数），影响延迟和性能 | 否 |
| `volume` | `float` | `0.3` | 音量（0.0-1.0），默认 30% | 否 |

**注意**:
- ⚠️ **禁止使用** `period_size`、`chunk_size` 等别名，统一使用 `buffer_size`
- 驱动特定的参数（如 ALSA 的 `device`）可以作为额外参数添加

### init() 方法签名

```python
def init(self, sample_rate: int = 50000) -> bool:
    """
    初始化音频驱动

    Args:
        sample_rate: 采样率，默认 50000 Hz (Ardens 标准)
                    libretro 核心会在运行时传入实际采样率

    Returns:
        初始化成功返回 True，失败返回 False
    """
```

### play_samples() 方法签名

```python
def play_samples(self, samples: np.ndarray) -> None:
    """
    播放音频采样

    Args:
        samples: 音频采样数据，numpy 数组格式
                - int16 格式：来自 libretro 核心，范围 [-32768, 32767]
                - float32 格式：归一化范围 [-1.0, 1.0]

    注意：
        - Ardens 输出 int16 格式，范围 [-SOUND_GAIN, +SOUND_GAIN] (SOUND_GAIN=2000)
        - 必须支持 int16 和 float32 两种格式
        - 必须正确处理音量控制和削波
    """
```

### close() 方法签名

```python
def close(self) -> None:
    """关闭音频驱动，释放资源"""
```

### set_volume() 方法签名（可选但推荐）

```python
def set_volume(self, volume: float) -> None:
    """
    设置音量

    Args:
        volume: 音量，0.0-1.0
    """
```

## 驱动对比表

### 参数对比

| 驱动名称 | sample_rate | channels | buffer_size | volume | 特有参数 |
|---------|-------------|----------|-------------|--------|---------|
| **NullAudioDriver** | ✅ 50000 | - | - | - | - |
| **AlsaAudioDriver** | ✅ 50000 | ✅ 1 | ✅ 2048 | ✅ 0.3 | `device` |
| **PyAudioDriver** | ✅ 50000 | ✅ 1 | ✅ 2048 | ✅ 0.3 | - |
| **PygameMixerDriver** | ✅ 50000 | ✅ 1 | ✅ 2048 | ✅ 0.3 | - |

### 方法对比

| 驱动名称 | init() | play_samples() | close() | set_volume() |
|---------|--------|----------------|---------|--------------|
| **NullAudioDriver** | ✅ | ✅ | ✅ | ❌ |
| **AlsaAudioDriver** | ✅ | ✅ | ✅ | ✅ |
| **PyAudioDriver** | ✅ | ✅ | ✅ | ✅ |
| **PygameMixerDriver** | ✅ | ✅ | ✅ | ✅ |

## 使用示例

### 统一的驱动实例化

所有驱动现在可以使用相同的参数进行实例化：

```python
# ALSA 驱动（树莓派推荐）
audio_driver = AlsaAudioDriver(
    sample_rate=50000,  # Ardens 标准: 16MHz / 320
    channels=1,         # 单声道（Arduboy 硬件）
    buffer_size=2048,   # 降低延迟
    volume=0.3          # 适中音量
)

# PyAudio 驱动（跨平台）
audio_driver = PyAudioDriver(
    sample_rate=50000,  # Ardens 标准: 16MHz / 320
    channels=1,         # 单声道（Arduboy 硬件）
    buffer_size=2048,   # 降低延迟
    volume=0.3          # 适中音量
)

# Pygame Mixer 驱动（游戏开发）
audio_driver = PygameMixerDriver(
    sample_rate=50000,  # Ardens 标准: 16MHz / 320
    channels=1,         # 单声道（Arduboy 硬件）
    buffer_size=2048,   # 降低延迟
    volume=0.3          # 适中音量
)

# 空驱动（无声模式）
audio_driver = NullAudioDriver()
```

### 驱动切换示例

由于接口统一，可以轻松切换驱动：

```python
from pyarduboy.drivers.audio import (
    AlsaAudioDriver,
    PyAudioDriver,
    PygameMixerDriver,
    NullAudioDriver
)
import platform

# 根据平台自动选择驱动
def create_audio_driver():
    """根据平台创建合适的音频驱动"""

    # 统一的配置参数
    config = {
        'sample_rate': 50000,
        'channels': 1,
        'buffer_size': 2048,
        'volume': 0.3
    }

    system = platform.system()

    if system == 'Linux':
        # 检测是否是树莓派
        try:
            with open('/proc/device-tree/model', 'r') as f:
                if 'Raspberry Pi' in f.read():
                    print("Detected Raspberry Pi, using ALSA driver")
                    return AlsaAudioDriver(device='default', **config)
        except:
            pass

        # 普通 Linux 桌面
        print("Using PyAudio driver")
        return PyAudioDriver(**config)

    elif system in ['Darwin', 'Windows']:
        print(f"Using PyAudio driver for {system}")
        return PyAudioDriver(**config)

    else:
        print(f"Unknown platform {system}, using null driver")
        return NullAudioDriver()

# 使用
driver = create_audio_driver()
driver.init()
```

## 音频格式处理规范

### 输入格式支持

所有驱动必须支持以下两种输入格式：

#### 1. int16 格式（来自 Ardens libretro 核心）

```python
# 范围: [-32768, 32767]
# Ardens 实际输出范围: [-SOUND_GAIN, +SOUND_GAIN] (SOUND_GAIN=2000)
samples = np.array([0, 1000, -1000, 2000, -2000], dtype=np.int16)
```

**处理方式**:
```python
if samples.dtype == np.int16:
    # int16 → float → 应用音量 → int16
    samples_float = samples.astype(np.float32) / 32768.0  # 归一化到 [-1, 1]
    if self.volume != 1.0:
        samples_float *= self.volume
    # 防止削波
    samples_float = np.clip(samples_float, -1.0, 1.0)
    samples_int16 = (samples_float * 32767).astype(np.int16)
```

#### 2. float32 格式（标准化格式）

```python
# 范围: [-1.0, 1.0]
samples = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
```

**处理方式**:
```python
else:  # float32
    samples_float = samples.astype(np.float32)
    if self.volume != 1.0:
        samples_float *= self.volume
    # 防止削波
    samples_float = np.clip(samples_float, -1.0, 1.0)
    samples_int16 = (samples_float * 32767).astype(np.int16)
```

### 声道转换规范

单声道转立体声（当 `self.channels == 2`）：

```python
if samples_int16.ndim == 1 and self.channels == 2:
    # 单声道转立体声：使用 repeat 优化内存访问
    samples_int16 = np.repeat(samples_int16, 2)
```

## 非阻塞设计要求

⚠️ **关键要求**: 所有驱动必须实现非阻塞播放，不能阻塞主循环。

### 推荐实现方式

1. **Callback 模式**（最佳）
   - 音频在独立线程中播放
   - 主循环只负责向缓冲队列添加数据
   - 示例: PyAudioDriver

2. **非阻塞 API**
   - 使用音频库的非阻塞模式
   - 避免等待音频播放完成
   - 示例: AlsaAudioDriver (`PCM_NONBLOCK`)

3. **智能队列管理**
   - 限制队列大小，避免延迟累积
   - 当队列满时，丢弃旧数据或跳过当前帧
   - 示例: PygameMixerDriver

### 错误示例（会阻塞）

```python
# ❌ 错误 - 会阻塞主循环
def play_samples(self, samples):
    sound = create_sound(samples)
    sound.play()
    sound.wait()  # 阻塞等待播放完成
```

### 正确示例（非阻塞）

```python
# ✅ 正确 - 非阻塞播放
def play_samples(self, samples):
    sound = create_sound(samples)

    if not self.channel.get_busy():
        self.channel.play(sound)  # 非阻塞播放
    elif not self.channel.get_queue():
        self.channel.queue(sound)  # 队列下一个
    # else: 队列已满，跳过避免延迟累积
```

## 错误处理规范

### 初始化失败

- 返回 `False`
- 打印清晰的错误信息
- 提供故障排除建议

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

### 播放错误

- 不崩溃，只打印错误
- 只在第一次错误时打印（避免刷屏）

```python
def play_samples(self, samples: np.ndarray) -> None:
    try:
        # 播放代码...
    except Exception as e:
        # 只在第一次错误时打印
        if not hasattr(self, '_error_printed'):
            print(f"Error playing samples: {e}")
            import traceback
            traceback.print_exc()
            self._error_printed = True
```

## 性能要求

### FPS 影响

- 音频驱动不应显著影响游戏 FPS
- 目标: 维持 60 FPS
- 测试: 使用 [性能测试脚本](README.md#性能测试)

### 延迟

| buffer_size | 延迟 (ms) @ 50kHz | 用途 |
|-------------|------------------|------|
| 512 | ~10ms | 低延迟（需要高 CPU） |
| 2048 | ~40ms | 平衡（推荐） |
| 4096 | ~80ms | 高质量（更稳定） |

计算公式: `延迟(ms) = (buffer_size / sample_rate) * 1000`

## 迁移指南

### 从旧接口迁移

如果你的代码使用了旧的参数名，请按照以下方式更新：

#### ❌ 旧代码（不推荐）

```python
# 旧的 ALSA 驱动接口
driver = AlsaAudioDriver(
    device='default',
    sample_rate=44100,  # 错误的采样率
    channels=2,         # 错误的声道数
    period_size=4096    # 旧参数名
)
```

#### ✅ 新代码（推荐）

```python
# 新的统一接口
driver = AlsaAudioDriver(
    device='default',
    sample_rate=50000,  # 正确: Ardens 标准
    channels=1,         # 正确: 单声道
    buffer_size=2048,   # 新参数名
    volume=0.3          # 新增: 音量控制
)
```

### PyAudio 驱动迁移

PyAudioDriver 之前使用了 `chunk_size` 参数，现已统一为 `buffer_size`：

```python
# ✅ 推荐: 使用 buffer_size
driver = PyAudioDriver(buffer_size=2048)

# ⚠️ 向后兼容: chunk_size 仍然支持，但已弃用
driver = PyAudioDriver(chunk_size=2048)  # 会自动映射到 buffer_size
```

## 测试清单

新驱动或修改后的驱动必须通过以下测试：

- [ ] 默认采样率为 50000 Hz
- [ ] 默认声道数为 1（单声道）
- [ ] 默认 buffer_size 为 2048
- [ ] 默认 volume 为 0.3
- [ ] `init()` 方法签名正确
- [ ] `play_samples()` 支持 int16 格式
- [ ] `play_samples()` 支持 float32 格式
- [ ] 音量控制工作正常
- [ ] 非阻塞播放（不影响 FPS）
- [ ] 错误处理正确（不崩溃）
- [ ] 文档完整（docstring）
- [ ] 通过单元测试
- [ ] 通过集成测试
- [ ] 通过性能测试（维持 60 FPS）

## 版本历史

### v1.0 (2025-12-04)

**初始版本**
- 统一采样率: 50000 Hz
- 统一声道数: 1（单声道）
- 统一参数名: `buffer_size`（替代 `period_size`、`chunk_size`）
- 新增音量控制: `volume` 参数和 `set_volume()` 方法
- 定义标准接口: `init()`, `play_samples()`, `close()`
- 音频格式支持: int16 和 float32
- 非阻塞设计要求
- 错误处理规范

## 参考资源

- [音频驱动开发指南](README.md)
- [Arduboy 官方规格](https://arduboy.com/)
- [Ardens 模拟器源码](https://github.com/tiberiusbrown/Ardens)

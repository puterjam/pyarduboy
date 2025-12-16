"""
LibRetro 桥接层

负责与 arduous_libretro 核心进行交互，封装 libretro.py 的复杂性
"""
import os
from pathlib import Path
import numpy as np
from typing import Optional

try:
    from libretro import SessionBuilder, ArrayVideoDriver, ArrayAudioDriver, DEFAULT
    from libretro.drivers.input import IterableInputDriver
    from libretro.api.input import JoypadState
except ImportError as e:
    raise ImportError(
        "libretro.py library not found. Please install it using: pip3 install libretro.py"
    ) from e


class LibretroBridge:
    """
    LibRetro 桥接类

    负责管理 libretro session 的生命周期，提供简化的接口给上层调用
    """

    def __init__(self, core_path: str, game_path: str, retro_path: Optional[str] = None):
        """
        初始化 LibRetro 桥接层

        Args:
            core_path: libretro 核心文件路径（.so 文件）
            game_path: 游戏 ROM 文件路径（.hex 文件）
            retro_path: 模拟器工作目录（可选，默认为当前工作目录）
                       存档将保存到 {retro_path}/saves/{游戏名}/ 目录下
        """
        if not os.path.exists(core_path):
            raise FileNotFoundError(f"Core file not found: {core_path}")

        if not os.path.exists(game_path):
            raise FileNotFoundError(f"Game file not found: {game_path}")

        self.core_path = core_path
        self.game_path = game_path

        # 模拟器工作目录设置
        if retro_path:
            self.retro_path = Path(retro_path)
        else:
            # 默认使用当前工作目录
            self.retro_path = Path.cwd()

        # 确保工作目录存在
        self.retro_path.mkdir(parents=True, exist_ok=True)

        # 存档目录结构: {retro_path}/saves/{游戏名}/
        game_name = Path(game_path).stem
        self.save_dir = self.retro_path / 'saves' / game_name
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # LibRetro 组件
        self.builder = None
        self.session = None
        self.video_driver = None
        self.audio_driver = None
        self.input_driver = None

        # 输入状态管理
        self.input_state = {
            'b': False,      # Button B
            'y': False,
            'select': False,
            'start': False,
            'up': False,
            'down': False,
            'left': False,
            'right': False,
            'a': False,      # Button A
            'x': False,
            'l': False,
            'r': False,
            'l2': False,
            'r2': False,
            'l3': False,
            'r3': False,
        }

        # 会话状态
        self._initialized = False
        self._running = False

        # 快速存档缓冲区（内存）
        self._quick_save_buffer: Optional[bytearray] = None

    def set_input_state(self, button_states: dict) -> None:
        """
        设置输入状态

        Args:
            button_states: 按键状态字典，例如：
                {
                    'up': True,
                    'down': False,
                    'a': True,
                    ...
                }
        """
        self.input_state.update(button_states)

    def _input_generator(self):
        """生成器函数，持续产生当前输入状态"""
        while self._running:
            yield JoypadState(**self.input_state)

    def initialize(self) -> bool:
        """
        初始化 LibRetro 会话

        Returns:
            初始化成功返回 True，失败返回 False
        """
        if self._initialized:
            return True

        try:
            # 创建 session builder
            self.builder = SessionBuilder()
            self.builder.with_core(self.core_path)
            self.builder.with_content(self.game_path)
            self.builder.with_content_driver(DEFAULT)

            # 启用性能优化（在某些 Python 版本可能失败，忽略错误继续）
            try:
                self.builder.with_jit_capable(True)
                self.builder.with_perf(DEFAULT)
                self.builder.with_timing(DEFAULT)
            except Exception as e:
                # 某些优化在新版本 Python 中可能不可用，但不影响核心功能
                print(f"Warning: Could not enable all optimizations: {e}")
                pass

            # 设置视频驱动
            self.video_driver = ArrayVideoDriver()
            self.builder.with_video(self.video_driver)

            # 设置音频驱动（在 Python 3.11+ 可能会失败）
            try:
                self.audio_driver = ArrayAudioDriver()
                self.builder.with_audio(self.audio_driver)
            except Exception as e:
                print(f"Warning: Could not initialize audio driver: {e}")
                print("Continuing without audio support...")
                self.audio_driver = None

            # 设置输入驱动
            self.input_driver = IterableInputDriver(lambda: self._input_generator())
            self.builder.with_input(self.input_driver)

            # 设置存档目录（重要：libretro 核心需要这个）
            try:
                from libretro.drivers.path import ExplicitPathDriver
                # 创建 ExplicitPathDriver 并设置存档目录
                # 注意：ExplicitPathDriver 有 bug，会对所有参数调用 makedirs，
                # 所以必须为所有路径提供有效值，不能使用 None
                # 为不需要的路径在工作目录下创建子目录
                system_dir = self.retro_path / 'system'
                assets_dir = self.retro_path / 'assets'
                saves_dir = self.retro_path / 'saves'  # saves 父目录
                playlist_dir = self.retro_path / 'playlists'

                path_driver = ExplicitPathDriver(
                    corepath=self.core_path,         # libretro 核心路径
                    system=str(system_dir),           # 系统目录（BIOS等）
                    assets=str(assets_dir),           # 核心资源目录
                    save=str(saves_dir),       # 存档父目录: {retro_path}/saves/
                    playlist=str(playlist_dir)        # 播放列表目录
                )
                self.builder.with_paths(path_driver)  # 注意：方法名是 with_paths (复数)
                print(f"✓ Retro working directory: {self.retro_path}")
                print(f"✓ Save directory: {self.save_dir}")
            except Exception as e:
                print(f"Warning: Could not configure save directory: {e}")
                import traceback
                traceback.print_exc()

            # 构建会话
            self.session = self.builder.build()

            self._initialized = True
            return True

        except Exception as e:
            print(f"Failed to initialize LibRetro bridge: {e}")
            import traceback
            traceback.print_exc()
            return False

    def start(self) -> bool:
        """
        启动 LibRetro 会话

        Returns:
            启动成功返回 True，失败返回 False
        """
        if not self._initialized:
            if not self.initialize():
                return False

        if self._running:
            return True

        try:
            # 进入会话上下文
            self.session.__enter__()
            self._running = True
            return True
        except Exception as e:
            print(f"Failed to start LibRetro session: {e}")
            return False

    def run_frame(self) -> None:
        """运行一帧模拟"""
        if not self._running:
            raise RuntimeError("Session not started. Call start() first.")

        self.session.run()

    def get_frame(self) -> Optional[np.ndarray]:
        """
        获取当前帧的图像数据

        Returns:
            numpy 数组，shape 为 (height, width, 3)，RGB 格式
            如果没有帧数据则返回 None
        """
        if not self.video_driver or not hasattr(self.video_driver, '_frame'):
            return None

        if self.video_driver._frame is None:
            return None

        try:
            # 重要：创建帧数据的完整副本，避免引用导致的拖影
            frame_data = bytes(self.video_driver._frame)
            width = self.video_driver._last_width
            height = self.video_driver._last_height
            pitch = self.video_driver._last_pitch
            pixel_format = getattr(self.video_driver, '_pixel_format', None)

            # 提取实际帧数据
            actual_size = height * pitch
            frame_bytes = frame_data[:actual_size]

            # 根据 pitch 判断像素格式
            bytes_per_pixel = pitch // width if width > 0 else 0
            # 根据像素格式处理
            if bytes_per_pixel == 2:
                # RGB565 格式 (每像素 2 字节)
                img_array = np.frombuffer(frame_bytes, dtype=np.uint16).reshape((height, width))

                # 转换 RGB565 到 RGB888
                r = ((img_array & 0xF800) >> 11) << 3
                g = ((img_array & 0x07E0) >> 5) << 2
                b = (img_array & 0x001F) << 3

                rgb_array = np.stack([r, g, b], axis=2).astype(np.uint8)
                # 返回副本，避免共享内存
                return rgb_array.copy()

            elif bytes_per_pixel == 1:
                # 8位索引色或灰度 (每像素 1 字节)
                img_array = np.frombuffer(frame_bytes, dtype=np.uint8).reshape((height, width))

                # 转换为 RGB (假设是灰度或单色)
                rgb_array = np.stack([img_array, img_array, img_array], axis=2)
                # 返回副本
                return rgb_array.copy()

            elif bytes_per_pixel == 4:
                # XRGB8888 或 ARGB8888 格式 (每像素 4 字节)
                # 使用 pitch 来正确 reshape (因为可能有 padding)
                img_array = np.frombuffer(frame_bytes, dtype=np.uint8).reshape((height, pitch))

                # 提取实际像素数据 (每行前 width*4 字节)
                img_array = img_array[:, :width*4].reshape((height, width, 4))

                # libretro pixel_format=1 通常是 XRGB8888 (小端序: B G R X)
                # 需要反转通道顺序: BGRX -> RGB
                rgb_array = img_array[:, :, [2, 1, 0]].copy()  # 提取并反转 BGR -> RGB，返回副本
                return rgb_array

            else:
                print(f"Unknown pixel format: bytes_per_pixel={bytes_per_pixel}, pitch={pitch}, width={width}")
                print(f"Data size: {len(frame_bytes)} bytes, height={height}")
                return None

        except Exception as e:
            print(f"Error getting frame: {e}")
            import traceback
            traceback.print_exc()
            return None

    def get_audio_samples(self) -> Optional[np.ndarray]:
        """
        获取音频采样数据

        Returns:
            numpy 数组 (int16 格式),包含音频采样
            如果没有音频数据则返回 None
        """
        if not self.audio_driver:
            return None

        try:
            # ArrayAudioDriver 将音频数据存储在 _buffer 属性中（array("h") 类型）
            if hasattr(self.audio_driver, '_buffer'):
                buffer = self.audio_driver._buffer

                if len(buffer) == 0:
                    return None

                # 转换 array("h") 到 numpy int16 数组 (保持原始格式,避免精度损失)
                samples_int16 = np.array(buffer, dtype=np.int16)

                # 可选: 写入 WAV 文件调试原始缓冲
                # self._write_audio_dump(samples_int16)

                # 调试: 打印音频样本数（每 60 帧打印一次）
                # if not hasattr(self, '_audio_frame_count'):
                #     self._audio_frame_count = 0
                # self._audio_frame_count += 1
                # if self._audio_frame_count % 60 == 0:
                #     print(f"[LibRetro Audio] Frame {self._audio_frame_count}: got {len(samples_int16)} samples")

                # 清空缓冲区（避免重复播放）
                self.audio_driver._buffer = self.audio_driver._buffer.__class__(self.audio_driver._buffer.typecode)

                return samples_int16 if len(samples_int16) > 0 else None

            return None
        except Exception as e:
            print(f"[Audio] Error getting samples: {e}")
            import traceback
            traceback.print_exc()
            return None

    def get_audio_sample_rate(self, default: int = 44100) -> int:
        try:
            if self.audio_driver and getattr(self.audio_driver, 'system_av_info', None):
                info = self.audio_driver.system_av_info
                if info and getattr(info, 'timing', None):
                    rate = getattr(info.timing, 'sample_rate', 0)
                    if rate:
                        return int(rate)
        except Exception:
            pass

        return default
    
    def stop(self) -> None:
        """停止 LibRetro 会话"""
        if self._running and self.session:
            try:
                self.session.__exit__(None, None, None)
            except Exception as e:
                print(f"Error stopping session: {e}")
            finally:
                self._running = False

    def reset(self) -> bool:
        """
        重置游戏（重新加载）

        Returns:
            重置成功返回 True，失败返回 False
        """
        print("Resetting game...")
        self.cleanup()

        # 重新初始化并启动
        if not self.initialize():
            print("Failed to reinitialize after reset")
            return False

        if not self.start():
            print("Failed to restart after reset")
            return False

        print("Game reset complete")
        return True

    def cleanup(self) -> None:
        """清理资源"""
        self.stop()
        self.session = None
        self.builder = None
        self.video_driver = None
        self.audio_driver = None
        self.input_driver = None
        self._initialized = False

    @property
    def is_running(self) -> bool:
        """检查会话是否正在运行"""
        return self._running

    @property
    def video_info(self) -> dict:
        """获取视频信息"""
        if self.video_driver:
            return {
                'width': getattr(self.video_driver, '_last_width', 0),
                'height': getattr(self.video_driver, '_last_height', 0),
                'pixel_format': getattr(self.video_driver, '_pixel_format', None),
            }
        return {}

    def __enter__(self):
        """上下文管理器入口"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        _ = exc_type, exc_val, exc_tb  # 未使用，但需要符合上下文管理器协议
        self.cleanup()

    # ==================== 存档功能 ====================

    def get_savestate_size(self) -> int:
        """
        获取存档所需的缓冲区大小

        Returns:
            存档大小（字节），如果不支持存档则返回 0
        """
        if not self._running or not self.session:
            return 0

        try:
            return self.session.core.serialize_size()
        except Exception as e:
            print(f"Error getting savestate size: {e}")
            return 0

    def save_state(self, slot: int = 0) -> bool:
        """
        保存游戏状态到指定槽位

        存档文件保存在 {retro_path}/saves/{游戏名}/slot{槽位号}.sav

        Args:
            slot: 存档槽位（0-99），默认为 0

        Returns:
            保存成功返回 True，失败返回 False
        """
        if not self._running or not self.session:
            print("Session not running, cannot save state")
            return False

        try:
            # 获取存档大小
            size = self.session.core.serialize_size()
            if size == 0:
                print("Core does not support save states")
                return False

            # 分配缓冲区
            buffer = bytearray(size)

            # 序列化游戏状态
            success = self.session.core.serialize(buffer)
            if not success:
                print("Failed to serialize game state")
                return False

            # 生成存档文件路径：self.save_dir 已经是 {retro_path}/saves/{游戏名}/
            save_file = self.save_dir / f"slot{slot}.sav"

            # 保存到文件
            with open(save_file, 'wb') as f:
                f.write(buffer)

            print(f"Game state saved to: {save_file}")
            print(f"  Size: {len(buffer)} bytes")
            return True

        except Exception as e:
            print(f"Error saving state: {e}")
            import traceback
            traceback.print_exc()
            return False

    def load_state(self, slot: int = 0) -> bool:
        """
        从指定槽位加载游戏状态

        存档文件从 {retro_path}/saves/{游戏名}/slot{槽位号}.sav 读取

        Args:
            slot: 存档槽位（0-99），默认为 0

        Returns:
            加载成功返回 True，失败返回 False
        """
        if not self._running or not self.session:
            print("Session not running, cannot load state")
            return False

        try:
            # 生成存档文件路径：self.save_dir 已经是 {retro_path}/saves/{游戏名}/
            save_file = self.save_dir / f"slot{slot}.sav"

            # 检查存档文件是否存在
            if not save_file.exists():
                print(f"Save file not found: {save_file}")
                return False

            # 读取存档数据
            with open(save_file, 'rb') as f:
                data = f.read()

            # 反序列化游戏状态
            success = self.session.core.unserialize(data)
            if not success:
                print("Failed to unserialize game state")
                return False

            print(f"Game state loaded from: {save_file}")
            print(f"  Size: {len(data)} bytes")
            return True

        except Exception as e:
            print(f"Error loading state: {e}")
            import traceback
            traceback.print_exc()
            return False

    def quick_save(self) -> bool:
        """
        快速保存到内存（用于即时回放/倒带）

        Returns:
            保存成功返回 True，失败返回 False
        """
        if not self._running or not self.session:
            return False

        try:
            size = self.session.core.serialize_size()
            if size == 0:
                return False

            self._quick_save_buffer = bytearray(size)
            success = self.session.core.serialize(self._quick_save_buffer)

            if success:
                print(f"Quick save: {len(self._quick_save_buffer)} bytes")
            return success

        except Exception as e:
            print(f"Error in quick save: {e}")
            return False

    def quick_load(self) -> bool:
        """
        快速加载（从内存）

        Returns:
            加载成功返回 True，失败返回 False
        """
        if not self._running or not self.session:
            return False

        if self._quick_save_buffer is None:
            print("No quick save available")
            return False

        try:
            success = self.session.core.unserialize(self._quick_save_buffer)
            if success:
                print("Quick load successful")
            return success

        except Exception as e:
            print(f"Error in quick load: {e}")
            return False

    def has_quick_save(self) -> bool:
        """
        检查是否有快速存档

        Returns:
            有快速存档返回 True，否则返回 False
        """
        return self._quick_save_buffer is not None

    def list_save_states(self) -> list[int]:
        """
        列出所有可用的存档槽位

        Returns:
            可用槽位列表，例如 [0, 1, 3] 表示槽位 0、1、3 有存档
        """
        if not self.save_dir.exists():
            return []

        slots = []
        for save_file in self.save_dir.glob("slot*.sav"):
            # 从文件名提取槽位号
            # 例如: "slot2.sav" -> 2
            try:
                slot_str = save_file.stem.replace('slot', '')
                slot = int(slot_str)
                slots.append(slot)
            except ValueError:
                continue

        return sorted(slots)

    def delete_save_state(self, slot: int = 0) -> bool:
        """
        删除指定槽位的存档

        Args:
            slot: 存档槽位（0-99）

        Returns:
            删除成功返回 True，失败返回 False
        """
        try:
            save_file = self.save_dir / f"slot{slot}.sav"

            if not save_file.exists():
                print(f"Save file not found: {save_file}")
                return False

            save_file.unlink()
            print(f"Deleted save state: {save_file}")
            return True

        except Exception as e:
            print(f"Error deleting save state: {e}")
            return False

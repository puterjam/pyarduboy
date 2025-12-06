"""
PyArduboy 核心类

提供高级 API 来运行 Arduboy 游戏，管理驱动插件
"""
import time
import platform
from pathlib import Path
from typing import Optional
from .libretro_bridge import LibretroBridge
from .drivers.base import VideoDriver, AudioDriver, InputDriver


class PyArduboy:
    """
    PyArduboy 核心类

    这是主要的接口类，用于加载和运行 Arduboy 游戏。
    支持插拔式驱动系统，可以自定义视频、音频和输入驱动。

    Example:
        >>> from pyarduboy import PyArduboy
        >>> from pyarduboy.drivers.video.luma_oled import LumaOLEDDriver
        >>>
        >>> # 自动判断核心路径(推荐)
        >>> arduboy = PyArduboy(game_path="./game.hex")
        >>>
        >>> # 或手动指定核心路径
        >>> arduboy = PyArduboy(
        ...     game_path="./game.hex",
        ...     core_path="./arduous_libretro.so"
        ... )
        >>> arduboy.set_video_driver(LumaOLEDDriver())
        >>> arduboy.run()
    """

    # Arduboy 默认配置
    SCREEN_WIDTH = 128
    SCREEN_HEIGHT = 64
    TARGET_FPS = 60

    # Game Boy 屏幕配置
    GB_SCREEN_WIDTH = 160
    GB_SCREEN_HEIGHT = 144

    # 支持的核心列表 (优先级顺序)
    SUPPORTED_CORES = ["ardens", "arduous", "gearboy"]

    @staticmethod
    def _get_lib_extension() -> str:
        """获取当前平台的库文件扩展名"""
        system = platform.system()
        if system == "Darwin":
            return "dylib"
        elif system == "Linux":
            return "so"
        elif system == "Windows":
            return "dll"
        else:
            raise ValueError(f"Unsupported platform: {system}")

    @classmethod
    def _find_core(cls, core_name: Optional[str] = None) -> str:
        """
        自动查找 libretro 核心文件

        查找顺序:
        1. 如果指定了 core_name，查找该核心
        2. 否则按优先级查找: arduous > ardens
        3. 查找目录: ./core/, ../core/, 当前目录

        Args:
            core_name: 核心名称 (arduous/ardens)，None 表示自动选择

        Returns:
            核心文件路径

        Raises:
            FileNotFoundError: 未找到任何核心
        """
        lib_ext = cls._get_lib_extension()

        # 搜索目录列表
        search_dirs = [
            Path.cwd() / "core",                    # ./core/
            Path(__file__).parent.parent / "core",  # pyarduboy 包的上级目录 /core
            Path.cwd(),                             # 当前目录
        ]

        # 确定要查找的核心列表
        if core_name:
            if core_name not in cls.SUPPORTED_CORES:
                raise ValueError(
                    f"Unknown core: {core_name}. "
                    f"Supported cores: {', '.join(cls.SUPPORTED_CORES)}"
                )
            cores_to_search = [core_name]
        else:
            cores_to_search = cls.SUPPORTED_CORES

        # 按优先级搜索
        for core in cores_to_search:
            core_filename = f"{core}_libretro.{lib_ext}"

            for search_dir in search_dirs:
                core_path = search_dir / core_filename

                if core_path.exists():
                    print(f"Using LibRetro core: {core} ({core_path})")
                    return str(core_path)

        # 未找到任何核心
        searched_paths = []
        for core in cores_to_search:
            core_filename = f"{core}_libretro.{lib_ext}"
            for search_dir in search_dirs:
                searched_paths.append(str(search_dir / core_filename))

        raise FileNotFoundError(
            f"No libretro core found. Searched paths:\n" +
            "\n".join(f"  - {p}" for p in searched_paths) +
            f"\n\nPlease download a core using:\n" +
            f"  python download_core.py {cores_to_search[0]}"
        )

    def __init__(
        self,
        game_path: str,
        core_path: Optional[str] = None,
        core_name: Optional[str] = None,
        target_fps: int = TARGET_FPS,
        retro_path: Optional[str] = None
    ):
        """
        初始化 PyArduboy

        Args:
            game_path: 游戏 ROM 文件路径
            core_path: libretro 核心文件路径(可选，默认根据系统自动判断)
            core_name: 核心名称 (arduous/ardens)，用于自动查找
                      如果同时指定 core_path 和 core_name，core_path 优先
            target_fps: 游戏逻辑帧率，默认 60 FPS
            retro_path: 模拟器工作目录（可选，默认为当前工作目录）
                       存档将保存到 {retro_path}/saves/{游戏名}/ 目录下
        """
        # 自动查找 core_path
        if core_path is None:
            core_path = self._find_core(core_name)

        self.core_path = core_path
        self.core_name = core_name  # 保存核心名称
        self.game_path = game_path
        self.target_fps = target_fps  # 游戏逻辑帧率

        # 提取核心名称（从路径中）
        detected_core = Path(core_path).stem.replace("_libretro", "")

        # 根据核心类型设置屏幕分辨率
        if detected_core == "gearboy" or core_name == "gearboy":
            self.screen_width = self.GB_SCREEN_WIDTH
            self.screen_height = self.GB_SCREEN_HEIGHT
        else:
            self.screen_width = self.SCREEN_WIDTH
            self.screen_height = self.SCREEN_HEIGHT

        # LibRetro 桥接层
        self.bridge = LibretroBridge(core_path, game_path, retro_path=retro_path)

        # 驱动插件
        self.video_driver: Optional[VideoDriver] = None
        self.audio_driver: Optional[AudioDriver] = None
        self.input_driver: Optional[InputDriver] = None

        # 运行状态
        self._running = False
        self._logic_frame_count = 0  # 游戏逻辑帧计数
        self._start_time = 0
        self._last_frame = None  # 用于检测帧是否更新

    def set_video_driver(self, driver: VideoDriver) -> None:
        """
        设置视频驱动

        Args:
            driver: VideoDriver 实例
        """
        self.video_driver = driver

    def set_audio_driver(self, driver: AudioDriver) -> None:
        """
        设置音频驱动

        Args:
            driver: AudioDriver 实例
        """
        self.audio_driver = driver

    def set_input_driver(self, driver: InputDriver) -> None:
        """
        设置输入驱动

        Args:
            driver: InputDriver 实例
        """
        self.input_driver = driver

    def initialize(self) -> bool:
        """
        初始化所有组件

        Returns:
            初始化成功返回 True，失败返回 False
        """
        # 初始化 LibRetro 桥接层
        if not self.bridge.initialize():
            print("Failed to initialize LibRetro bridge")
            return False

        # 初始化视频驱动
        if self.video_driver:
            if not self.video_driver.init(self.screen_width, self.screen_height):
                print("Failed to initialize video driver")
                return False

        # 初始化音频驱动（优先使用核心报告的真实采样率）
        if self.audio_driver:
            audio_sample_rate = self.bridge.get_audio_sample_rate(default=44100)
            if not self.audio_driver.init(sample_rate=audio_sample_rate):
                print("Failed to initialize audio driver")
                return False

        # 初始化输入驱动
        if self.input_driver:
            if not self.input_driver.init():
                print("Failed to initialize input driver")
                return False

        return True

    def run(self, max_frames: Optional[int] = None) -> None:
        """
        运行游戏主循环

        Args:
            max_frames: 最大运行帧数,None 表示无限运行
        """
        if not self.initialize():
            print("Initialization failed")
            return

        if not self.bridge.start():
            print("Failed to start emulation")
            return

        self._running = True
        self._frame_count = 0
        self._logic_frame_count = 0
        self._start_time = time.time()
        self._logic_accumulator = 0.0  # 累积的逻辑时间
        last_time = time.perf_counter()

        # 提取核心名称
        core_name = Path(self.core_path).stem.replace("_libretro", "")

        print(f"Starting emulation...")
        print(f"Core: {core_name}")
        print(f"Game: {self.game_path}")
        print(f"Screen Resolution: {self.screen_width}x{self.screen_height}")
        print(f"Game Logic FPS: {self.target_fps}")

        if self.video_driver:
            print(f"Video Driver: {self.video_driver.__class__.__name__}")
        if self.audio_driver:
            print(f"Audio Driver: {self.audio_driver.__class__.__name__}")
        if self.input_driver:
            print(f"Input Driver: {self.input_driver.__class__.__name__}")
        print()

        try:
            while self._running:
                # 检查视频驱动是否已关闭（例如用户关闭了窗口）
                if self.video_driver and hasattr(self.video_driver, 'is_running'):
                    if not self.video_driver.is_running:
                        print("\nVideo window closed by user")
                        break

                frame_start = time.perf_counter()
                # 基于真实时间累加逻辑时间，避免固定步长误差
                dt = frame_start - last_time
                last_time = frame_start
                self._logic_accumulator += dt

                # 计算本帧应该执行几次游戏逻辑
                logic_frame_time = 1.0 / self.target_fps

                # 处理输入 (每次显示帧都处理)
                if self.input_driver:
                    input_state = self.input_driver.poll()

                    # 检查 reset 按钮
                    if input_state.get('reset', False):
                        print("\n" + "="*60)
                        print("Reset button pressed!")
                        print("="*60 + "\n")
                        # 重置游戏
                        if self.bridge.reset():
                            self._frame_count = 0
                            self._logic_frame_count = 0
                            self._logic_accumulator = 0.0
                            self._start_time = time.time()
                        continue

                    # 转换输入状态到 LibRetro 格式
                    self.bridge.set_input_state(self._convert_input_state(input_state))

                # 执行游戏逻辑 (根据累积时间决定执行次数)
                logic_executed = False
                while self._logic_accumulator >= logic_frame_time:
                    self.bridge.run_frame()
                    self._logic_frame_count += 1
                    self._logic_accumulator -= logic_frame_time
                    logic_executed = True

                    # 获取并缓存新的游戏帧
                    if self.video_driver:
                        frame = self.bridge.get_frame()
                        if frame is not None:
                            self._last_frame = frame

                # ✅ 关键: 每个显示帧获取一次音频(不管执行了几次逻辑帧)
                # 音频会在 libretro buffer 中自然累积,一次性取出播放
                # 避免在 while 循环内重复获取导致音频数据翻倍
                if self.audio_driver and logic_executed:
                    samples = self.bridge.get_audio_samples()
                    if samples is not None:
                        self.audio_driver.play_samples(samples)

                self._frame_count += 1

                # 渲染视频 (每次显示帧都渲染,重复显示上一个逻辑帧)
                if self.video_driver and self._last_frame is not None:
                    self.video_driver.render(self._last_frame)

                # 检查是否达到最大帧数
                # if max_frames and self._logic_frame_count >= max_frames:
                #     break

                # 帧率控制
                frame_elapsed = time.perf_counter() - frame_start
                frame_time = logic_frame_time / 5 # 5 倍逻辑帧时间
                if frame_elapsed < frame_time:
                    time.sleep(frame_time - frame_elapsed)

                # 打印统计信息（每 500 core render 帧）
                # if self._frame_count % 500 == 0:
                #     self._print_stats()

        except KeyboardInterrupt:
            print("\n\nStopping emulation...")
        except Exception as e:
            print(f"\nError during emulation: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()

    def _convert_input_state(self, input_state: dict) -> dict:
        """
        将输入驱动的状态转换为 LibRetro 输入状态

        Args:
            input_state: 输入驱动返回的状态字典

        Returns:
            LibRetro 格式的输入状态
        """
        return {
            'up': input_state.get('up', False),
            'down': input_state.get('down', False),
            'left': input_state.get('left', False),
            'right': input_state.get('right', False),
            'a': input_state.get('a', False),
            'b': input_state.get('b', False),
            'select': input_state.get('select', False),
            'start': input_state.get('start', False),
            'x': False,
            'y': False,
            'l': False,
            'r': False,
            'l2': False,
            'r2': False,
            'l3': False,
            'r3': False,
        }

    def _print_stats(self) -> None:
        """打印运行统计信息"""
        elapsed = time.time() - self._start_time
        display_fps = self._frame_count / elapsed if elapsed > 0 else 0
        logic_fps = self._logic_frame_count / elapsed if elapsed > 0 else 0
        print(f"Display Frame {self._frame_count}: Core Loop FPS={display_fps:.1f}, Game Logic FPS={logic_fps:.1f}")

    def stop(self) -> None:
        """停止运行"""
        self._running = False

    def cleanup(self) -> None:
        """清理资源"""
        self._running = False

        # 关闭驱动
        if self.video_driver:
            self.video_driver.close()
        if self.audio_driver:
            self.audio_driver.close()
        if self.input_driver:
            self.input_driver.close()

        # 清理 LibRetro 桥接层
        self.bridge.cleanup()

        print("Emulation stopped.")

    @property
    def is_running(self) -> bool:
        """检查是否正在运行"""
        return self._running

    @property
    def frame_count(self) -> int:
        """获取当前帧数"""
        return self._frame_count

    @property
    def fps(self) -> float:
        """获取当前实际帧率"""
        if not self._start_time or self._frame_count == 0:
            return 0.0
        elapsed = time.time() - self._start_time
        return self._frame_count / elapsed if elapsed > 0 else 0.0

    def __enter__(self):
        """上下文管理器入口"""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        _ = exc_type, exc_val, exc_tb  # 未使用，但需要符合上下文管理器协议
        self.cleanup()

    # ==================== 存档功能 ====================

    def save_state(self, slot: int = 0) -> bool:
        """
        保存游戏状态到指定槽位

        存档文件保存在 {retro_path}/saves/{游戏名}/slot{槽位号}.sav

        Args:
            slot: 存档槽位（0-99），默认为 0

        Returns:
            保存成功返回 True，失败返回 False
        """
        return self.bridge.save_state(slot)

    def load_state(self, slot: int = 0) -> bool:
        """
        从指定槽位加载游戏状态

        Args:
            slot: 存档槽位（0-99），默认为 0

        Returns:
            加载成功返回 True，失败返回 False
        """
        return self.bridge.load_state(slot)

    def quick_save(self) -> bool:
        """
        快速保存到内存（用于即时回放/倒带）

        Returns:
            保存成功返回 True，失败返回 False
        """
        return self.bridge.quick_save()

    def quick_load(self) -> bool:
        """
        快速加载（从内存）

        Returns:
            加载成功返回 True，失败返回 False
        """
        return self.bridge.quick_load()

    def has_quick_save(self) -> bool:
        """
        检查是否有快速存档

        Returns:
            有快速存档返回 True，否则返回 False
        """
        return self.bridge.has_quick_save()

    def list_save_states(self) -> list[int]:
        """
        列出所有可用的存档槽位

        Returns:
            可用槽位列表，例如 [0, 1, 3] 表示槽位 0、1、3 有存档
        """
        return self.bridge.list_save_states()

    def delete_save_state(self, slot: int = 0) -> bool:
        """
        删除指定槽位的存档

        Args:
            slot: 存档槽位（0-99）

        Returns:
            删除成功返回 True，失败返回 False
        """
        return self.bridge.delete_save_state(slot)

    @property
    def retro_directory(self) -> Path:
        """获取模拟器工作目录路径"""
        return self.bridge.retro_path

    @property
    def save_directory(self) -> Path:
        """获取当前游戏的存档目录路径 ({retro_path}/saves/{游戏名}/)"""
        return self.bridge.save_dir

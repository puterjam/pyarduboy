"""
Evdev 键盘输入驱动

使用 evdev 库直接读取物理键盘事件
适合在无头系统或需要直接硬件访问的场景
"""
import threading
from typing import Optional
from .base import InputDriver

try:
    from evdev import InputDevice, categorize, ecodes, list_devices
    EVDEV_AVAILABLE = True
except ImportError:
    EVDEV_AVAILABLE = False
    ecodes = None  # 防止后续引用错误


def _get_default_key_map():
    """获取默认按键映射（延迟初始化）"""
    if not EVDEV_AVAILABLE:
        return {}
    return {
        ecodes.KEY_W: 'up',
        ecodes.KEY_S: 'down',
        ecodes.KEY_A: 'left',
        ecodes.KEY_D: 'right',
        ecodes.KEY_G: 'select',
        ecodes.KEY_H: 'start',
        ecodes.KEY_K: 'a',
        ecodes.KEY_J: 'b',
        ecodes.KEY_R: 'reset',  # R 键重置游戏
    }


class EvdevKeyboardDriver(InputDriver):
    """
    Evdev 键盘输入驱动

    直接从 /dev/input/ 读取键盘事件
    支持多个键同时按下，更符合游戏操作习惯

    默认按键映射（使用键盘扫描码）：
        W - 上
        S - 下
        A - 左
        D - 右
        J - A 按钮
        K - B 按钮
        R - Reset（重新加载游戏）

    Args:
        device_path: 键盘设备路径，如 '/dev/input/event0'
                    如果为 None，则自动检测第一个键盘设备
        key_map: 自定义按键映射（使用 evdev 键码）
    """

    def __init__(self, device_path: Optional[str] = None, key_map: Optional[dict] = None):
        """
        初始化 Evdev 键盘驱动

        Args:
            device_path: 键盘设备路径，None 表示自动检测
            key_map: 自定义按键映射
        """
        super().__init__()

        if not EVDEV_AVAILABLE:
            raise ImportError(
                "evdev library not found. Please install it:\n"
                "  sudo pip3 install evdev\n"
                "or\n"
                "  sudo apt-get install python3-evdev"
            )

        self.device_path = device_path
        self.key_map = key_map or _get_default_key_map()
        self.device: Optional[InputDevice] = None
        self.button_state = {
            'up': False,
            'down': False,
            'left': False,
            'right': False,
            'select': False,
            'start': False,
            'a': False,
            'b': False,
            'reset': False,
        }
        self.thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def _find_keyboard_device(self) -> Optional[str]:
        """
        自动查找键盘设备
        优先选择带 LED 支持的主键盘设备（避免选择副设备）

        Returns:
            键盘设备路径，如果未找到则返回 None
        """
        devices = [InputDevice(path) for path in list_devices()]

        # 候选键盘设备
        keyboard_devices = []

        for device in devices:
            # 检查设备是否支持键盘事件
            capabilities = device.capabilities()
            if ecodes.EV_KEY in capabilities:
                # 检查是否有键盘按键（不只是鼠标按键）
                keys = capabilities[ecodes.EV_KEY]
                # 如果有字母键，认为是键盘
                if ecodes.KEY_A in keys or ecodes.KEY_W in keys:
                    has_led = ecodes.EV_LED in capabilities
                    keyboard_devices.append((device, has_led))

        if not keyboard_devices:
            return None

        # 优先选择有 LED 支持的设备（通常是主键盘）
        for device, has_led in keyboard_devices:
            if has_led:
                print(f"Auto-detected keyboard: {device.name} at {device.path}")
                print(f"  (Primary keyboard with LED support)")
                return device.path

        # 如果没有 LED 设备，选择第一个
        device = keyboard_devices[0][0]
        print(f"Auto-detected keyboard: {device.name} at {device.path}")
        return device.path

    def init(self) -> bool:
        """初始化键盘设备"""
        try:
            # 如果没有指定设备路径，自动查找
            if self.device_path is None:
                self.device_path = self._find_keyboard_device()

            if self.device_path is None:
                print("Error: No keyboard device found")
                print("Available devices:")
                for path in list_devices():
                    dev = InputDevice(path)
                    print(f"  {path}: {dev.name}")
                return False

            # 打开设备
            self.device = InputDevice(self.device_path)
            print(f"Opened keyboard device: {self.device.name}")

            # 打印按键映射表
            print(f"Key mapping:")
            for key_code, button in self.key_map.items():
                print(f"  Key code {key_code} -> {button}")

            # 尝试独占设备（可选，防止输入泄漏到其他程序）
            try:
                self.device.grab()
                print("Device grabbed (exclusive access)")
            except Exception as e:
                print(f"Warning: Could not grab device: {e}")
                print("Input may leak to other programs")

            # 启动事件监听线程
            self._running = True
            self.thread = threading.Thread(target=self._event_loop, daemon=True)
            self.thread.start()

            return True

        except Exception as e:
            print(f"Failed to initialize keyboard device: {e}")
            return False

    def _event_loop(self):
        """事件监听循环（在单独线程中运行）"""
        print("Event loop started, waiting for keyboard input...")

        try:
            for event in self.device.read_loop():
                if not self._running:
                    break

                # 只处理按键事件，跳过其他类型（高效过滤）
                if event.type == ecodes.EV_KEY:
                    key_code = event.code

                    # 只处理映射的按键，减少不必要的操作
                    if key_code in self.key_map:
                        # event.value: 0=释放, 1=按下, 2=保持按下
                        key_state = event.value
                        button = self.key_map[key_code]
                        is_pressed = (key_state == 1 or key_state == 2)

                        # 更新按键状态（线程安全，快速返回）
                        with self._lock:
                            self.button_state[button] = is_pressed

        except OSError:
            # 设备被关闭（正常退出）
            pass
        except Exception as e:
            print(f"Error in event loop: {e}")
            import traceback
            traceback.print_exc()

    def poll(self) -> dict:
        """
        轮询输入状态

        Returns:
            按键状态字典
        """
        with self._lock:
            return self.button_state.copy()

    def close(self) -> None:
        """关闭键盘设备"""
        self._running = False

        if self.thread:
            self.thread.join(timeout=1.0)

        if self.device:
            try:
                self.device.ungrab()
            except Exception:
                pass

            try:
                self.device.close()
            except Exception:
                pass

            self.device = None


def _get_arrow_key_map():
    """获取方向键映射（延迟初始化）"""
    if not EVDEV_AVAILABLE:
        return {}
    return {
        ecodes.KEY_UP: 'up',
        ecodes.KEY_DOWN: 'down',
        ecodes.KEY_LEFT: 'left',
        ecodes.KEY_RIGHT: 'right',
        ecodes.KEY_X: 'a',
        ecodes.KEY_Z: 'b',
        ecodes.KEY_G: 'select',
        ecodes.KEY_H: 'start',
        ecodes.KEY_R: 'reset',  # R 键重置游戏
    }


class EvdevKeyboardDriverCustom(EvdevKeyboardDriver):
    """
    可自定义按键的 Evdev 键盘驱动

    示例：使用方向键和 Z/X 按钮
    """

    def __init__(self, device_path: Optional[str] = None):
        # 使用方向键 + Z/X 的按键映射
        custom_map = _get_arrow_key_map()
        super().__init__(device_path=device_path, key_map=custom_map)

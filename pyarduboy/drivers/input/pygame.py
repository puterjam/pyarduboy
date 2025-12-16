"""
Pygame 键盘输入驱动

使用 pygame 事件系统读取键盘输入
适合桌面环境（macOS、Windows、Linux GUI）
"""
from typing import Dict
from .base import InputDriver

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False


class PygameKeyboardDriver(InputDriver):
    """
    Pygame 键盘输入驱动

    使用 pygame 事件系统处理键盘输入
    支持多个键同时按下

    默认按键映射：
        方向键 - 上下左右
        Z - A 按钮
        X - B 按钮
        R - Reset（重新加载游戏）
        ESC - 退出（由 pygame 视频驱动处理）

    Args:
        key_map: 自定义按键映射字典（可选）
    """

    def __init__(self, key_map: Dict[int, str] = None):
        super().__init__()

        if not PYGAME_AVAILABLE:
            raise ImportError("pygame is not installed. Please run: pip install pygame")

        # 默认按键映射
        self.key_map = key_map or {
            pygame.K_UP: 'up',
            pygame.K_DOWN: 'down',
            pygame.K_LEFT: 'left',
            pygame.K_RIGHT: 'right',
            pygame.K_z: 'a',
            pygame.K_x: 'b',
            pygame.K_g: 'select',
            pygame.K_h: 'start',
            pygame.K_r: 'reset',
        }

        # 当前按键状态
        self.key_state = {
            'up': False,
            'down': False,
            'left': False,
            'right': False,
            'a': False,
            'b': False,
            'select': False,
            'start': False,
            'reset': False,
        }

        # 上一帧的 reset 按键状态（用于边沿触发检测）
        self._last_reset_state = False

        self._running = False

    def init(self) -> bool:
        """
        初始化输入驱动

        Returns:
            初始化成功返回 True
        """
        if not PYGAME_AVAILABLE:
            print("Error: pygame is not available")
            return False

        # pygame 应该已经由视频驱动初始化了
        if not pygame.get_init():
            print("Warning: pygame not initialized, initializing now...")
            pygame.init()

        self._running = True
        return True

    def poll(self) -> dict:
        """
        轮询输入状态

        注意：pygame 事件由视频驱动处理（PygameDriver.render）
        这里只返回键盘按键状态

        Returns:
            按键状态字典
        """
        if not self._running:
            return self.key_state

        # 获取当前按键状态（pygame.key.get_pressed）
        keys = pygame.key.get_pressed()

        # 更新按键状态
        for key_code, action in self.key_map.items():
            if action in self.key_state:
                if action == 'reset':
                    # Reset 按键使用边沿触发（只在按下瞬间触发一次）
                    current_state = keys[key_code]
                    # 检测上升沿：从 False 变为 True
                    self.key_state['reset'] = current_state and not self._last_reset_state
                    self._last_reset_state = current_state
                else:
                    # 其他按键使用电平触发（持续按下时持续为 True）
                    self.key_state[action] = keys[key_code]

        return self.key_state.copy()

    def close(self) -> None:
        """关闭输入驱动"""
        self._running = False

    @property
    def is_running(self) -> bool:
        """检查驱动是否正在运行"""
        return self._running


class PygameKeyboardDriverWASD(PygameKeyboardDriver):
    """
    Pygame 键盘输入驱动（WASD 模式）

    使用 WASD 作为方向键，适合习惯 FPS 游戏的玩家
    """

    def __init__(self, **kwargs):
        # 创建 WASD 按键映射
        import pygame
        key_map = {
            pygame.K_w: 'up',
            pygame.K_s: 'down',
            pygame.K_a: 'left',
            pygame.K_d: 'right',
            pygame.K_g: 'select',
            pygame.K_h: 'start',
            pygame.K_j: 'a',
            pygame.K_k: 'b',
            pygame.K_r: 'reset',
        }
        super().__init__(key_map=key_map)

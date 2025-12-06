"""
Null 输入驱动

空输入驱动，用于测试或不需要输入的场景
"""
from .base import InputDriver


class NullInputDriver(InputDriver):
    """
    Null 输入驱动

    不提供任何输入，所有按键状态始终为 False
    用于测试或演示模式
    """

    def __init__(self):
        super().__init__()
        self._running = False

        # 空按键状态
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

    def init(self) -> bool:
        """
        初始化输入驱动

        Returns:
            总是返回 True
        """
        self._running = True
        return True

    def poll(self) -> dict:
        """
        轮询输入状态

        Returns:
            空按键状态字典（所有按键都是 False）
        """
        return self.key_state.copy()

    def close(self) -> None:
        """关闭输入驱动"""
        self._running = False

    @property
    def is_running(self) -> bool:
        """检查驱动是否正在运行"""
        return self._running

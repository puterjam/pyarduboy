"""
输入驱动基类
"""
from ..base import InputDriver as BaseInputDriver


class InputDriver(BaseInputDriver):
    """输入驱动基类，提供一些通用工具方法"""

    def __init__(self):
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

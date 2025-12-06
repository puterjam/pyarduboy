"""
输入驱动插件
"""
from .base import InputDriver

try:
    from .evdev import EvdevKeyboardDriver, EvdevKeyboardDriverCustom
    __all__ = [
        "InputDriver",
        "EvdevKeyboardDriver",
        "EvdevKeyboardDriverCustom",
    ]
except ImportError:
    # evdev 未安装时只导出基础驱动
    __all__ = [
        "InputDriver",
    ]

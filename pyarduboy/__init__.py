"""
PyArduboy - Python library for running Arduboy games on Raspberry Pi

This library provides a clean interface to run Arduboy games using the
arduous_libretro core with pluggable driver system for video, audio, and input.

Usage:
    from pyarduboy import PyArduboy
    from pyarduboy.drivers.video.luma_oled import LumaOLEDDriver

    # Create emulator instance
    arduboy = PyArduboy(
        core_path="./arduous_libretro.so",
        game_path="./game.hex"
    )

    # Set drivers
    arduboy.set_video_driver(LumaOLEDDriver())

    # Run the game
    arduboy.run()
"""

__version__ = "0.1.0"
__author__ = "PuterJam@gmail.com"

# Patch dataclasses to work with libretro.py ctypes structures
import dataclasses

def _patch_dataclasses_for_libretro():
    """
    libretro.py 的 ctypes 结构没有文档字符串,dataclasses 会尝试读取
    inspect.signature 导致 ValueError。这里提前为这些结构补一个 docstring,
    让 dataclasses 跳过签名注入逻辑。
    """
    flag_name = "_libretro_doc_patch"
    if getattr(dataclasses, flag_name, False):
        return

    original = dataclasses._process_class  # type: ignore[attr-defined]

    def patched_process_class(cls, *args, **kwargs):
        module_name = getattr(cls, "__module__", "")
        if not getattr(cls, "__doc__", None) and module_name.startswith("libretro."):
            cls.__doc__ = f"{cls.__name__} structure"
        return original(cls, *args, **kwargs)

    dataclasses._process_class = patched_process_class  # type: ignore[assignment]
    setattr(dataclasses, flag_name, True)

_patch_dataclasses_for_libretro()

from .core import PyArduboy
from .libretro_bridge import LibretroBridge

# Import base driver classes for easier access
from .drivers.base import VideoDriver, AudioDriver, InputDriver

__all__ = [
    "PyArduboy",
    "LibretroBridge",
    "VideoDriver",
    "AudioDriver",
    "InputDriver",
]
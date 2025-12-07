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

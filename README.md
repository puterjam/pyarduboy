# PyArduboy

Python library for running Arduboy games using libretro cores.

## Overview

PyArduboy provides a flexible, driver-based architecture for emulating Arduboy games on various platforms. It supports multiple display, audio, and input drivers, making it adaptable to different hardware configurations.

## Features

- **Multiple Video Drivers**: Support for Pygame, Luma.OLED, and null (headless) displays
- **Multiple Audio Drivers**: PyAudio, Pygame Mixer, ALSA, or null audio output
- **Multiple Input Drivers**: Pygame keyboard/joystick, evdev, or null input
- **Libretro Core Integration**: Compatible with Arduboy libretro cores
- **Modular Architecture**: Easy to extend with custom drivers

## Installation

```bash
pip install -e .
```

## Usage

```python
from pyarduboy import ArduboyCore

# Initialize with default drivers
core = ArduboyCore(
    rom_path="path/to/game.arduboy",
    core_path="path/to/libretro_core.so"
)

# Run the emulator
core.run()
```

## Driver System

PyArduboy uses a modular driver system for video, audio, and input:

- **Video Drivers**: `pygame`, `luma`, `null`
- **Audio Drivers**: `pyaudio`, `pygame_mixer`, `alsa`, `null`
- **Input Drivers**: `pygame`, `evdev`, `null`

## Requirements

- Python 3.7+
- NumPy
- Driver-specific dependencies (pygame, luma.oled, pyaudio, etc.)

## License

MIT License

## Part of

This library is used by [pyarduboy-runner](https://github.com/puterjam/pyarduboy-runner) - a complete Arduboy emulator for Raspberry Pi and other platforms.

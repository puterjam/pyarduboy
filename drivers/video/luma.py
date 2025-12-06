"""
Luma.OLED 视频驱动

使用 luma.oled 库驱动 SSD1305/SSD1306 等 OLED 显示屏
"""
import numpy as np
from PIL import Image
from .base import VideoDriver


class LumaOLEDDriver(VideoDriver):
    """
    Luma.OLED 驱动实现 (使用 luma 标准 SPI,受频率限制)

    支持常见的 SPI OLED 显示屏，如 SSD1305, SSD1306, SSD1309
    注意: 此驱动仅支持 luma 预定义的 SPI 频率 (8, 16, 20, 24... MHz)
    如需任意频率,请使用 LumaCustomDriver

    Args:
        device_type: 设备类型 'ssd1309', 'ssd1306' 等
        width: 显示宽度，默认 128
        height: 显示高度，默认 64（Arduboy 标准）
        spi_speed_hz: SPI 频率(Hz), 必须是 luma 支持的值 (8MHz, 16MHz 等)
        gpio_DC: DC 引脚号 (BCM 编号)
        gpio_RST: RST 引脚号 (BCM 编号)
        rotate: 旋转角度（0, 1, 2, 3），默认 2
        dither_mode: 抖动模式 'none', 'floyd', 'threshold'
    """

    def __init__(
        self,
        device_type: str = 'ssd1309',
        width: int = 128,
        height: int = 64,
        spi_speed_hz: int = 8000000,
        gpio_DC: int = 25,
        gpio_RST: int = 27,
        rotate: int = 0,
        dither_mode: str = 'none'
    ):
        super().__init__()
        self.device_type = device_type
        self.display_width = width
        self.display_height = height
        self.spi_speed_hz = spi_speed_hz
        self.gpio_DC = gpio_DC
        self.gpio_RST = gpio_RST
        self.rotate = rotate
        self.dither_mode = dither_mode
        self.device = None
        self._gpio = None

    def init(self, width: int, height: int) -> bool:
        """
        初始化 OLED 显示设备

        Args:
            width: Arduboy 屏幕宽度（128）
            height: Arduboy 屏幕高度（64）

        Returns:
            初始化成功返回 True
        """
        self._width = width
        self._height = height

        try:
            import RPi.GPIO as GPIO
            from luma.core.interface.serial import spi
            from luma.oled.device import ssd1309, ssd1306

            # 设置 GPIO
            GPIO.setmode(GPIO.BCM)
            self._gpio = GPIO

            # 创建 luma 标准 SPI 接口 (受频率限制)
            serial = spi(
                port=0,
                device=0,
                gpio_DC=self.gpio_DC,
                gpio_RST=self.gpio_RST,
                gpio=GPIO,
                bus_speed_hz=self.spi_speed_hz
            )

            # 创建对应的 OLED 设备
            if self.device_type == 'ssd1309':
                self.device = ssd1309(
                    serial,
                    width=self.display_width,
                    height=self.display_height,
                    rotate=self.rotate
                )
            elif self.device_type == 'ssd1306':
                self.device = ssd1306(
                    serial,
                    width=self.display_width,
                    height=self.display_height,
                    rotate=self.rotate
                )
            else:
                print(f"Unsupported device type: {self.device_type}")
                return False

            print(f"✓ Luma OLED initialized: {self.display_width}x{self.display_height} @ {self.spi_speed_hz/1e6:.0f} MHz")
            self._running = True
            return True

        except ImportError as e:
            print(f"Failed to import dependencies: {e}")
            print("Please install: pip3 install luma.oled")
            return False
        except Exception as e:
            print(f"Failed to initialize OLED: {e}")
            import traceback
            traceback.print_exc()
            return False

    def render(self, frame_buffer: np.ndarray) -> None:
        """
        渲染一帧到 OLED 显示屏

        Args:
            frame_buffer: 帧缓冲区，shape 为 (height, width, 3)，RGB 格式
        """
        if not self._running or self.device is None:
            return

        try:
            # 转换为 PIL Image
            img = Image.fromarray(frame_buffer, 'RGB')


            img = img.convert('L')
            bw_img = img.convert('L').convert('1')

            # 显示到 OLED
            self.device.display(bw_img)

        except Exception as e:
            # 只在第一次错误时打印，避免刷屏
            if not hasattr(self, '_error_printed'):
                print(f"Error rendering frame: {e}")
                self._error_printed = True

    def close(self) -> None:
        """关闭显示设备"""
        self._running = False
        if self.device:
            try:
                self.device.clear()
            except Exception:
                pass
        if self._gpio:
            try:
                self._gpio.cleanup()
            except Exception:
                pass


class LumaOLED32Driver(LumaOLEDDriver):
    """
    专门用于 128x32 OLED 显示屏的驱动

    直接设置高度为 32
    """

    def __init__(self, **kwargs):
        kwargs['height'] = 32
        super().__init__(**kwargs)

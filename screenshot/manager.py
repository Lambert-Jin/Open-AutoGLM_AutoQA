"""截图管理器"""

from __future__ import annotations

import base64
import logging
import os

from config.settings import Screenshot
from device import Device

logger = logging.getLogger(__name__)


class ScreenshotManager:
    """截图获取与持久化"""

    def __init__(self, device: Device | None = None):
        self._device = device

    def capture(self, device: Device | None = None) -> Screenshot:
        """从设备截图，返回 Screenshot 对象

        Args:
            device: 可选，覆盖初始化时绑定的设备
        """
        dev = device or self._device
        if dev is None:
            raise RuntimeError("未绑定设备，请在初始化时传入 device 或在 capture() 时传入")

        ds = dev.screenshot()
        logger.debug("截图完成: %dx%d", ds.width, ds.height)
        return Screenshot(
            base64=ds.base64_data,
            width=ds.width,
            height=ds.height,
        )

    def from_file(self, path: str) -> Screenshot:
        """从本地图片文件加载为 Screenshot"""
        with open(path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()
        width, height = self._get_image_size(img_b64)
        return Screenshot(base64=img_b64, width=width, height=height)

    def save(self, screenshot: Screenshot, path: str) -> str:
        """将 Screenshot 保存为图片文件"""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            f.write(base64.b64decode(screenshot.base64))
        logger.debug("截图已保存: %s", path)
        return path

    @staticmethod
    def _get_image_size(img_b64: str) -> tuple[int, int]:
        """从图片 base64 数据解析宽高"""
        import struct

        header = base64.b64decode(img_b64[:200])

        # PNG: IHDR chunk at offset 16
        if header[:8] == b"\x89PNG\r\n\x1a\n":
            width, height = struct.unpack(">II", header[16:24])
            return width, height

        # JPEG: 扫描 SOF marker
        data = base64.b64decode(img_b64)
        i = 2
        while i < len(data) - 9:
            if data[i] != 0xFF:
                break
            marker = data[i + 1]
            if marker in (0xC0, 0xC1, 0xC2):  # SOF0, SOF1, SOF2
                height, width = struct.unpack(">HH", data[i + 5 : i + 9])
                return width, height
            length = struct.unpack(">H", data[i + 2 : i + 4])[0]
            i += 2 + length

        return 0, 0

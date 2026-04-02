"""实时帧缓冲"""

from __future__ import annotations

import base64
import io
from threading import Condition

from PIL import Image, ImageDraw

from device.base import DeviceScreenshot


class FrameHub:
    """保存最近一帧，供 MJPEG 输出"""

    def __init__(self):
        self._condition = Condition()
        self._frame_id = 0
        self._jpeg_data = self._create_placeholder()
        self._source = "placeholder"

    @staticmethod
    def _create_placeholder() -> bytes:
        image = Image.new("RGB", (540, 960), color=(20, 24, 28))
        draw = ImageDraw.Draw(image)
        draw.text((36, 48), "Waiting for scrcpy frame...", fill=(230, 230, 230))
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=85)
        return buf.getvalue()

    def publish_image(self, image: Image.Image, *, source: str = "scrcpy") -> int:
        rgb = image.convert("RGB")
        buf = io.BytesIO()
        rgb.save(buf, format="JPEG", quality=85)
        data = buf.getvalue()
        with self._condition:
            self._frame_id += 1
            self._jpeg_data = data
            self._source = source
            self._condition.notify_all()
            return self._frame_id

    def publish_screenshot(self, screenshot: DeviceScreenshot, *, source: str = "device") -> int:
        image = Image.open(io.BytesIO(base64.b64decode(screenshot.base64_data)))
        return self.publish_image(image, source=source)

    def wait_for_frame(self, last_frame_id: int, timeout: float = 1.0) -> tuple[int, bytes, str]:
        with self._condition:
            if self._frame_id <= last_frame_id:
                self._condition.wait(timeout=timeout)
            return self._frame_id, self._jpeg_data, self._source

    def snapshot(self) -> tuple[int, bytes, str]:
        with self._condition:
            return self._frame_id, self._jpeg_data, self._source

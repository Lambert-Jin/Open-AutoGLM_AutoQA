"""动作截图标注"""

from __future__ import annotations

import base64
import io
import os

from PIL import Image, ImageDraw, ImageFont

from device.base import DeviceScreenshot
from monitor.models import OverlayAction


class ScreenshotAnnotator:
    """在截图上绘制动作标记"""

    def __init__(self):
        self._font = ImageFont.load_default()

    def render(self, screenshot: DeviceScreenshot, overlay: OverlayAction, phase: str) -> bytes:
        image = Image.open(io.BytesIO(base64.b64decode(screenshot.base64_data))).convert("RGBA")
        draw = ImageDraw.Draw(image, "RGBA")

        self._draw_title(draw, image.size, f"{phase}: {overlay.label}")
        self._draw_overlay(draw, image.size, overlay)

        out = io.BytesIO()
        image.convert("RGB").save(out, format="PNG")
        return out.getvalue()

    def save(self, screenshot: DeviceScreenshot, overlay: OverlayAction, phase: str, path: str) -> str:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            f.write(self.render(screenshot, overlay, phase))
        return path

    def _draw_title(self, draw: ImageDraw.ImageDraw, image_size: tuple[int, int], text: str) -> None:
        x, y = 18, 18
        bbox = draw.textbbox((x, y), text, font=self._font)
        padding_x = 14
        padding_y = 10
        draw.rounded_rectangle(
            (
                bbox[0] - padding_x,
                bbox[1] - padding_y,
                bbox[2] + padding_x,
                bbox[3] + padding_y,
            ),
            radius=12,
            fill=(12, 12, 12, 180),
        )
        draw.text((x, y), text, fill=(255, 255, 255, 255), font=self._font)

    def _draw_overlay(self, draw: ImageDraw.ImageDraw, image_size: tuple[int, int], overlay: OverlayAction) -> None:
        width, height = image_size
        radius = max(16, min(width, height) // 40)

        if overlay.x_norm is not None and overlay.y_norm is not None:
            x = int(overlay.x_norm * width)
            y = int(overlay.y_norm * height)
            self._draw_focus(draw, x, y, radius)

            if overlay.end_x_norm is not None and overlay.end_y_norm is not None:
                end_x = int(overlay.end_x_norm * width)
                end_y = int(overlay.end_y_norm * height)
                self._draw_arrow(draw, x, y, end_x, end_y, radius)
        elif overlay.type == "scroll":
            self._draw_scroll_hint(draw, width, height, overlay.direction or "down", radius)

        if overlay.text and overlay.type in {"type", "launch"}:
            self._draw_caption(draw, width, height, overlay.text)

    def _draw_focus(self, draw: ImageDraw.ImageDraw, x: int, y: int, radius: int) -> None:
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=(255, 90, 54, 255), width=6)
        draw.ellipse((x - radius * 2, y - radius * 2, x + radius * 2, y + radius * 2), outline=(255, 90, 54, 120), width=4)
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=(255, 90, 54, 255))

    def _draw_arrow(self, draw: ImageDraw.ImageDraw, x1: int, y1: int, x2: int, y2: int, radius: int) -> None:
        draw.line((x1, y1, x2, y2), fill=(255, 90, 54, 220), width=8)
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            return
        length = max((dx * dx + dy * dy) ** 0.5, 1)
        ux = dx / length
        uy = dy / length
        arrow = radius * 1.6
        left = (x2 - ux * arrow - uy * arrow * 0.7, y2 - uy * arrow + ux * arrow * 0.7)
        right = (x2 - ux * arrow + uy * arrow * 0.7, y2 - uy * arrow - ux * arrow * 0.7)
        draw.polygon([(x2, y2), left, right], fill=(255, 90, 54, 240))

    def _draw_scroll_hint(
        self,
        draw: ImageDraw.ImageDraw,
        width: int,
        height: int,
        direction: str,
        radius: int,
    ) -> None:
        center_x = width // 2
        center_y = height // 2
        distance = min(width, height) // 5
        offsets = {
            "up": (0, -distance),
            "down": (0, distance),
            "left": (-distance, 0),
            "right": (distance, 0),
        }
        dx, dy = offsets.get(direction, (0, distance))
        self._draw_focus(draw, center_x, center_y, radius)
        self._draw_arrow(draw, center_x, center_y, center_x + dx, center_y + dy, radius)

    def _draw_caption(self, draw: ImageDraw.ImageDraw, width: int, height: int, text: str) -> None:
        box_width = min(width - 36, max(160, width // 2))
        x = 18
        y = max(80, height - 120)
        bbox = draw.textbbox((x + 12, y + 10), text, font=self._font)
        draw.rounded_rectangle((x, y, x + box_width, bbox[3] + 18), radius=12, fill=(12, 12, 12, 180))
        draw.text((x + 12, y + 10), text, fill=(255, 255, 255, 255), font=self._font)

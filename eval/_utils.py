"""评测系统公共工具"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def load_image(path: str) -> dict | None:
    """加载截图文件为 OpenAI 图片消息块，自动检测 MIME 类型"""
    if not path:
        return None
    try:
        import base64
        from providers._utils import guess_mime_type

        with open(path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()
        mime = guess_mime_type(img_b64)
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{img_b64}"},
        }
    except FileNotFoundError:
        logger.warning("截图文件不存在: %s", path)
        return None

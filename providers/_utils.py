"""Provider 公共工具"""

from __future__ import annotations

import base64


def guess_mime_type(image_base64: str) -> str:
    """通过 base64 解码后的文件头字节判断图片格式"""
    header = base64.b64decode(image_base64[:32])

    if header[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if header[:2] == b"\xff\xd8":
        return "image/jpeg"
    if header[:4] == b"RIFF" and header[8:12] == b"WEBP":
        return "image/webp"

    # 默认按 PNG 处理
    return "image/png"

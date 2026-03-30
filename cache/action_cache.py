"""Action 缓存核心逻辑：lookup / store / validate"""

from __future__ import annotations

import base64
import hashlib
import io
import logging
import time
from dataclasses import dataclass

from cache.embedder import Embedder, cosine_similarity
from cache.store import CacheStore, CacheEntry
from device.base import DeviceScreenshot

logger = logging.getLogger(__name__)


@dataclass
class CachedAction:
    """缓存命中结果"""
    entry: CacheEntry
    similarity: float

    def to_action_params(self) -> dict:
        """转换为执行参数"""
        params = {
            "action_type": self.entry.action_type,
            "x": self.entry.x,
            "y": self.entry.y,
        }
        if self.entry.end_x is not None:
            params["end_x"] = self.entry.end_x
        if self.entry.end_y is not None:
            params["end_y"] = self.entry.end_y
        return params


class ActionCache:
    """Action 缓存：三层查找（精确过滤 → 语义匹配 → 视觉验证）"""

    # pHash 区域大小（坐标周围的截取范围，相对于截图尺寸的比例）
    REGION_RATIO = 0.1  # 截图宽度的 10%

    def __init__(
        self,
        embedder: Embedder,
        store: CacheStore,
        similarity_threshold: float = 0.85,
        region_similarity_threshold: float = 0.8,
        ttl_days: int = 30,
    ):
        self.embedder = embedder
        self.store = store
        self.similarity_threshold = similarity_threshold
        self.region_similarity_threshold = region_similarity_threshold
        self.ttl_days = ttl_days

    def lookup(
        self,
        cache_key: str,
        app: str,
        activity: str,
        screenshot: DeviceScreenshot,
    ) -> CachedAction | None:
        """
        缓存查找：
        1. 精确过滤：app
        2. 语义匹配：embedding cosine similarity
        3. 视觉验证：对比坐标区域 pHash
        """
        # 1. 精确过滤
        candidates = self.store.query(app=app)
        if not candidates:
            return None

        # 2. 语义匹配
        query_embedding = self.embedder.encode(cache_key)

        best_match = None
        best_score = 0.0
        for entry in candidates:
            score = cosine_similarity(query_embedding, entry.key_embedding)
            if score > self.similarity_threshold and score > best_score:
                best_match = entry
                best_score = score

        if not best_match:
            return None

        # 3. 视觉验证
        current_region_hash = self._compute_region_hash(
            screenshot, best_match.x, best_match.y
        )
        if not self._region_similar(current_region_hash, best_match.region_hash):
            logger.info("缓存视觉验证失败，坐标区域已变化: %s", cache_key)
            return None

        return CachedAction(entry=best_match, similarity=best_score)

    def store_action(
        self,
        cache_key: str,
        app: str,
        activity: str,
        action_type: str,
        x: int,
        y: int,
        screenshot: DeviceScreenshot,
        end_x: int | None = None,
        end_y: int | None = None,
    ):
        """执行成功后存入缓存"""
        embedding = self.embedder.encode(cache_key)
        region_hash = self._compute_region_hash(screenshot, x, y)
        self.store.upsert(CacheEntry(
            cache_key=cache_key,
            key_embedding=embedding,
            app=app,
            activity=activity,
            action_type=action_type,
            x=x,
            y=y,
            end_x=end_x,
            end_y=end_y,
            region_hash=region_hash,
            created_at=time.time(),
        ))

    def record_hit(self, entry: CacheEntry):
        """记录缓存命中"""
        self.store.update_hit(entry.cache_key, entry.app, entry.activity)

    def cleanup(self):
        """清理过期条目"""
        self.store.cleanup_expired(self.ttl_days)

    def _compute_region_hash(
        self, screenshot: DeviceScreenshot, x: int, y: int
    ) -> str:
        """
        计算坐标周围区域的 pHash。
        坐标是归一化的 (0-999)，需要映射到实际像素再截取区域。
        """
        try:
            from PIL import Image

            # 解码截图
            img_data = base64.b64decode(screenshot.base64_data)
            img = Image.open(io.BytesIO(img_data))

            # 归一化坐标 → 实际像素
            px = int(x / 1000 * img.width)
            py = int(y / 1000 * img.height)

            # 截取区域
            region_size = int(img.width * self.REGION_RATIO)
            half = region_size // 2
            left = max(0, px - half)
            top = max(0, py - half)
            right = min(img.width, px + half)
            bottom = min(img.height, py + half)

            # 保证裁剪区域有效
            if right <= left or bottom <= top:
                return ""

            region = img.crop((left, top, right, bottom))

            # 计算 pHash：缩小到 8x8 灰度 → 计算均值 → 生成 64 位哈希
            return self._phash(region)
        except Exception as e:
            logger.warning("计算 region hash 失败: %s", e)
            return ""

    @staticmethod
    def _phash(image) -> str:
        """简化版 pHash：8x8 灰度 → 均值哈希"""
        from PIL import Image

        small = image.resize((8, 8), Image.Resampling.LANCZOS).convert("L")
        pixels = list(small.getdata())
        avg = sum(pixels) / len(pixels)
        bits = "".join("1" if p > avg else "0" for p in pixels)
        # 转为 16 位十六进制字符串
        return hex(int(bits, 2))[2:].zfill(16)

    def _region_similar(self, hash1: str, hash2: str) -> bool:
        """比较两个 pHash 的汉明距离"""
        if not hash1 or not hash2:
            # 无法计算时默认不通过（避免假阳性）
            return False
        distance = self._hamming_distance(hash1, hash2)
        # 64 位哈希，汉明距离越小越相似
        # threshold 0.8 → 最多允许 64 * 0.2 = 12.8 位不同
        max_distance = int(64 * (1 - self.region_similarity_threshold))
        return distance <= max_distance

    @staticmethod
    def _hamming_distance(hash1: str, hash2: str) -> int:
        """计算两个十六进制哈希的汉明距离"""
        try:
            val1 = int(hash1, 16)
            val2 = int(hash2, 16)
            return bin(val1 ^ val2).count("1")
        except ValueError:
            return 64  # 无法比较，返回最大距离

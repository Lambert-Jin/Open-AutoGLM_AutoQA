"""Embedding 服务：本地 sentence-transformers，零 API 成本"""

from __future__ import annotations

import logging
import math

logger = logging.getLogger(__name__)


class Embedder:
    """本地 embedding，基于 sentence-transformers"""

    MODEL_NAME = "all-MiniLM-L6-v2"  # ~80MB，384 维，短文本效果好

    def __init__(self):
        self._model = None
        self._cache: dict[str, list[float]] = {}

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info("加载 embedding 模型: %s", self.MODEL_NAME)
            self._model = SentenceTransformer(self.MODEL_NAME)

    def encode(self, text: str) -> list[float]:
        """编码单个文本为向量"""
        if text in self._cache:
            return self._cache[text]
        self._load_model()
        embedding = self._model.encode(text).tolist()
        self._cache[text] = embedding
        return embedding

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        """批量编码"""
        self._load_model()
        results = self._model.encode(texts).tolist()
        for text, emb in zip(texts, results):
            self._cache[text] = emb
        return results


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """计算两个向量的余弦相似度"""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

"""全局配置数据类"""

from __future__ import annotations

from dataclasses import dataclass, field

from utils.text import resolve_env_vars as _resolve_env_vars


@dataclass
class VLMConfig:
    """VLM 配置，支持多 provider"""
    provider: str = "gemini"        # "qwen" | "gemini"
    base_url: str = ""              # Qwen 需要，Gemini 不需要
    api_key: str = "${GEMINI_API_KEY}"
    model: str = "gemini-3-pro-image-preview"
    temperature: float = 0.1        # 断言场景用低温度
    max_tokens: int = 1000

    def __post_init__(self):
        if self.api_key:
            self.api_key = _resolve_env_vars(self.api_key)
        if self.base_url:
            self.base_url = _resolve_env_vars(self.base_url)


@dataclass
class ActionModelConfig:
    """操作模型配置（支持多 provider）"""
    provider: str = "autoglm"       # "autoglm" | 将来扩展 "uitars" 等
    base_url: str = "${AUTOGLM_BASE_URL}"
    api_key: str = "${AUTOGLM_API_KEY}"
    model: str = "autoglm-phone"
    max_tokens: int = 3000
    temperature: float = 0.1
    lang: str = "cn"                # "cn" | "en"
    custom_rules: list[str] = field(default_factory=list)

    def __post_init__(self):
        if self.api_key:
            self.api_key = _resolve_env_vars(self.api_key)
        if self.base_url:
            self.base_url = _resolve_env_vars(self.base_url)


@dataclass
class LLMConfig:
    """LLM 配置（Planner 规划 + Optimizer 指令优化共用）"""
    provider: str = "gemini"
    base_url: str = ""              # Qwen 等 OpenAI 兼容 API 需要
    api_key: str = "${GEMINI_API_KEY}"
    model: str = "gemini-3.1-pro-preview"
    temperature: float = 0.3
    max_tokens: int = 2000

    def __post_init__(self):
        if self.api_key:
            self.api_key = _resolve_env_vars(self.api_key)
        if self.base_url:
            self.base_url = _resolve_env_vars(self.base_url)


@dataclass
class DeviceConfig:
    """设备配置"""
    device_type: str = "adb"        # "adb" | "hdc" | "ios"
    device_id: str | None = None    # None 为自动检测


@dataclass
class Screenshot:
    """一次截图的数据"""
    base64: str                     # 图片 base64 编码
    width: int = 0
    height: int = 0
    timestamp: float = 0.0         # time.time()
    id: str = ""                    # 唯一标识，用于报告关联

    def __post_init__(self):
        import time
        import uuid
        if not self.timestamp:
            self.timestamp = time.time()
        if not self.id:
            self.id = uuid.uuid4().hex[:8]


@dataclass
class CacheConfig:
    """缓存配置"""
    enabled: bool = True
    similarity_threshold: float = 0.85
    region_similarity_threshold: float = 0.8
    ttl_days: int = 30
    db_path: str = ".cache/action_cache.db"


@dataclass
class AssertResult:
    """断言结果"""
    passed: bool
    reason: str
    confidence: float = 1.0
    retried: bool = False           # 是否经过容错重试后通过
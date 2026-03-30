"""模型适配器"""

from __future__ import annotations

from executor.model_protocol import ActionModel
from executor.models.autoglm import AutoGLMConfig, AutoGLMModel


def create_action_model(
    provider: str = "autoglm",
    **kwargs,
) -> ActionModel:
    """模型适配器工厂。扩展新模型时只需新增分支。"""
    if provider == "autoglm":
        config = AutoGLMConfig(
            base_url=kwargs.get("base_url", AutoGLMConfig.base_url),
            api_key=kwargs.get("api_key", ""),
            model=kwargs.get("model", AutoGLMConfig.model),
            max_tokens=kwargs.get("max_tokens", AutoGLMConfig.max_tokens),
            temperature=kwargs.get("temperature", AutoGLMConfig.temperature),
            lang=kwargs.get("lang", "cn"),
            custom_rules=kwargs.get("custom_rules", []),
        )
        return AutoGLMModel(config)
    else:
        raise ValueError(f"不支持的模型 provider: {provider}")

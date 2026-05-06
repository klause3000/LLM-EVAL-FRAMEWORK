"""Judge client 工厂 + 公共导出。"""

import os

import yaml

from .base import BaseJudgeClient, JudgeScore
from .anthropic_client import AnthropicJudgeClient
from .openai_client import OpenAIJudgeClient
from .ollama_client import OllamaJudgeClient


def build_judge_client(config_path: str = "config/judge_config.yaml") -> BaseJudgeClient:
    """根据 judge_config.yaml 构造对应的 judge client。"""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    j = cfg["judge"]
    provider: str = j["provider"]
    model: str = j["model"]
    # api_key 直接填值优先；api_key_env 填环境变量名（更安全，推荐生产使用）
    api_key: str | None = j.get("api_key") or (
        os.environ.get(j["api_key_env"]) if j.get("api_key_env") else None
    )
    max_retries: int = j.get("max_retries", 3)
    timeout: int = j.get("timeout", 60)

    if provider == "anthropic":
        return AnthropicJudgeClient(
            model=model, api_key=api_key, max_retries=max_retries, timeout=timeout
        )
    if provider == "openai":
        base_url: str | None = j.get("base_url")  # 可选，指向兼容 OpenAI 协议的第三方服务
        return OpenAIJudgeClient(
            model=model, api_key=api_key, base_url=base_url, max_retries=max_retries, timeout=timeout
        )
    if provider == "ollama":
        base_url: str = j.get("base_url", "http://localhost:11434/v1")
        return OllamaJudgeClient(model=model, base_url=base_url, timeout=timeout)

    raise ValueError(f"不支持的 judge provider: {provider}，可选: anthropic | openai | ollama")


__all__ = [
    "BaseJudgeClient",
    "JudgeScore",
    "AnthropicJudgeClient",
    "OpenAIJudgeClient",
    "OllamaJudgeClient",
    "build_judge_client",
]

"""Ollama judge client（复用 OpenAI 兼容接口）。"""

from .openai_client import OpenAIJudgeClient

OLLAMA_BASE_URL = "http://localhost:11434/v1"


class OllamaJudgeClient(OpenAIJudgeClient):
    """通过 ollama 的 OpenAI 兼容接口调用本地强模型做 judge。"""

    def __init__(
        self,
        model: str,
        base_url: str = OLLAMA_BASE_URL,
        timeout: int = 120,
        **kwargs,
    ):
        super().__init__(
            model=model,
            api_key="ollama",   # ollama 不校验 key
            base_url=base_url,
            timeout=timeout,
            **kwargs,
        )

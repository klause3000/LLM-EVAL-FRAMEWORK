"""
Ollama 模型管理器：自动拉取、检查可用性、返回 OpenAI 兼容 client。
ollama 提供 OpenAI 兼容接口（/v1），直接用 openai.OpenAI(base_url=...) 调用，无需专用 SDK。
"""

import logging
import subprocess
import time
from typing import Optional

import yaml
from openai import OpenAI

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = "http://localhost:11434/v1"


def load_models(config_path: str = "config/models.yaml") -> list[dict]:
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("models", [])


def is_model_available(model_name: str) -> bool:
    """检查 ollama 本地是否已存在该模型。"""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True, text=True, timeout=10
        )
        return model_name in result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.warning("检查模型可用性失败: %s", e)
        return False


def pull_model(model_name: str, timeout: int = 600) -> bool:
    """
    拉取 ollama 模型，若已存在则跳过。
    返回 True 表示成功（含跳过），False 表示失败。
    """
    if is_model_available(model_name):
        logger.info("模型已存在，跳过拉取: %s", model_name)
        return True

    logger.info("开始拉取模型: %s", model_name)
    try:
        result = subprocess.run(
            ["ollama", "pull", model_name],
            timeout=timeout,
            text=True
        )
        if result.returncode == 0:
            logger.info("模型拉取成功: %s", model_name)
            return True
        else:
            logger.error("模型拉取失败（returncode=%d）: %s", result.returncode, model_name)
            return False
    except subprocess.TimeoutExpired:
        logger.error("模型拉取超时（%ds）: %s", timeout, model_name)
        return False
    except FileNotFoundError:
        logger.error("ollama 命令未找到，请先安装 ollama")
        return False


def pull_all_models(config_path: str = "config/models.yaml") -> dict[str, bool]:
    """读取 models.yaml，批量拉取所有模型，返回 {model_name: success} 映射。"""
    models = load_models(config_path)
    results = {}
    for model in models:
        name = model["name"]
        results[name] = pull_model(name)
    return results


def get_client(model_name: Optional[str] = None) -> OpenAI:
    """
    返回指向本地 ollama 的 OpenAI 兼容 client。
    model_name 仅作日志用，不影响 client 构造。
    """
    return OpenAI(
        base_url=OLLAMA_BASE_URL,
        api_key="ollama",  # ollama 不校验 key，任意非空字符串均可
    )


def chat(model_name: str, prompt: str, system: str = "", timeout: int = 120) -> str:
    """
    向 ollama 模型发送单次对话请求，返回回答文本。
    """
    client = get_client(model_name)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        timeout=timeout,
    )
    return response.choices[0].message.content or ""


def wait_for_ollama(retries: int = 5, interval: int = 3) -> bool:
    """等待 ollama 服务就绪（用于 CI 环境启动延迟）。"""
    for i in range(retries):
        try:
            client = get_client()
            client.models.list()
            logger.info("ollama 服务已就绪")
            return True
        except Exception:
            logger.info("等待 ollama 服务（%d/%d）...", i + 1, retries)
            time.sleep(interval)
    logger.error("ollama 服务未就绪")
    return False

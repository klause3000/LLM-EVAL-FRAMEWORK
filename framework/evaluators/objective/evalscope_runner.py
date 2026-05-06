"""
客观评测运行器：通过 evalscope 对标准数据集（ceval/mmlu/cmmlu/gsm8k 等）评测。

evalscope 通过 OpenAI-compatible API 与 ollama 通信，
需将 model_args.api_base 指向本地 ollama 端点。
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = "http://localhost:11434/v1"


def load_objective_config(config_path: str = "config/objective_config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_objective_eval(
    model_name: str,
    model_version: str = "",
    config_path: str = "config/objective_config.yaml",
) -> dict[str, dict]:
    """
    对指定 ollama 模型运行客观评测。

    :param model_name: ollama 模型名，如 "qwen2.5:7b"
    :param model_version: 版本标识（仅用于入库，不影响评测）
    :param config_path: objective_config.yaml 路径
    :return: {dataset_name: {"score": float|None, "num_samples": int, "raw": dict, "error": str|None}}
    """
    cfg = load_objective_config(config_path)
    datasets_cfg: list[dict] = cfg.get("datasets", [])
    enabled: list[str] = cfg.get("enabled_datasets") or []
    evalscope_params: dict = cfg.get("evalscope_params", {})
    limit: Optional[int] = evalscope_params.get("limit")

    if enabled:
        datasets_cfg = [d for d in datasets_cfg if d["name"] in enabled]

    all_results: dict[str, dict] = {}
    for ds in datasets_cfg:
        ds_name: str = ds["name"]
        num_shots: int = ds.get("shots", 0)
        logger.info("开始客观评测: model=%s dataset=%s shots=%d", model_name, ds_name, num_shots)
        try:
            result = _run_single_dataset(model_name, ds_name, num_shots, limit)
        except Exception as e:
            logger.error("客观评测失败: dataset=%s error=%s", ds_name, e)
            result = {"score": None, "num_samples": 0, "raw": {}, "error": str(e)}
        all_results[ds_name] = result

    return all_results


_CACHE_ERROR_KEYWORDS = ("couldn't find cache", "response ended prematurely", "no (supported) data files")


def _run_single_dataset(
    model_name: str,
    dataset_name: str,
    num_shots: int,
    limit: Optional[int],
) -> dict:
    """调用 evalscope 运行单个数据集，返回标准化结果。遇到 cache/网络错误自动重试一次。"""
    try:
        from evalscope.run import run_task
        from evalscope.config import TaskConfig
    except ImportError as e:
        raise ImportError(
            "evalscope 未安装，请执行: pip install evalscope>=0.6"
        ) from e

    def _build_cfg(force_redownload: bool) -> "TaskConfig":
        args: dict = {"few_shot_num": num_shots}
        if force_redownload:
            args["force_redownload"] = True
        return TaskConfig(
            model=model_name,
            datasets=[dataset_name],
            dataset_args={dataset_name: args},
            eval_type="openai_api",
            api_url=OLLAMA_BASE_URL,
            api_key="ollama",
            limit=limit,
            use_cache=False,
        )

    try:
        output = run_task(task_cfg=_build_cfg(force_redownload=False))
        return _parse_evalscope_output(output, dataset_name)
    except Exception as first_err:
        err_str = str(first_err).lower()
        if any(kw in err_str for kw in _CACHE_ERROR_KEYWORDS):
            logger.warning(
                "数据集 %s 首次加载失败（cache/网络问题），5s 后重试（force_redownload=True）: %s",
                dataset_name, first_err,
            )
            time.sleep(5)
            try:
                output = run_task(task_cfg=_build_cfg(force_redownload=True))
                return _parse_evalscope_output(output, dataset_name)
            except Exception as retry_err:
                raise retry_err from first_err
        raise


def _parse_evalscope_output(output: Any, dataset_name: str) -> dict:
    """
    将 evalscope 输出解析为统一格式。
    evalscope 1.6 返回 {dataset_name: Report对象}，Report 有 .score 和 .metrics 属性。
    """
    try:
        # output 是 dict: {dataset_name: Report}
        report_obj = output.get(dataset_name) if isinstance(output, dict) else output

        score: Optional[float] = None
        num_samples: int = 0

        if report_obj is not None:
            # Report 对象直接有 .score 属性
            if hasattr(report_obj, "score"):
                score = float(report_obj.score)
            # 从 metrics 取样本数
            if hasattr(report_obj, "metrics") and report_obj.metrics:
                num_samples = int(report_obj.metrics[0].num)

        return {"score": score, "num_samples": num_samples, "raw": str(report_obj), "error": None}
    except Exception as e:
        logger.warning("解析 evalscope 输出失败: %s", e)
        return {"score": None, "num_samples": 0, "raw": str(output), "error": str(e)}

"""生成质量评测：调用被测模型 → 计算 BLEU/ROUGE/召回率。"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from framework.model_manager.ollama_manager import chat as ollama_chat
from .metrics import GenerationMetrics, compute_metrics

logger = logging.getLogger(__name__)


@dataclass
class GenerationCaseResult:
    case_id: str
    dataset_name: str
    question: str
    model_answer: str
    reference_answer: str
    metrics: GenerationMetrics
    tags: list[str] = field(default_factory=list)


def run_generation_eval(
    model_name: str,
    cases: list[dict],
    dataset_name: str,
) -> list[GenerationCaseResult]:
    """
    对指定模型在给定 cases 上运行生成质量评测。

    :param model_name: ollama 模型名
    :param cases:      数据集 case 列表（需含 reference_answer）
    :param dataset_name: 数据集名称
    :return: 每条 case 的结果列表
    """
    results: list[GenerationCaseResult] = []

    for case in cases:
        case_id: str = case["id"]
        question: str = case["question"]
        reference: str = case.get("reference_answer") or ""
        tags: list[str] = case.get("tags") or []

        if not reference:
            logger.warning("[%s] case %s 无 reference_answer，跳过生成质量评测", model_name, case_id)
            continue

        logger.info("[%s] 生成质量评测 case: %s", model_name, case_id)

        try:
            model_answer = ollama_chat(model_name, question)
        except Exception as e:
            logger.error("获取模型回答失败 case=%s: %s", case_id, e)
            model_answer = ""

        metrics = compute_metrics(model_answer, reference)
        logger.debug(
            "case=%s bleu=%.1f rouge1=%.3f rougeL=%.3f recall=%.3f",
            case_id, metrics.bleu, metrics.rouge1, metrics.rougeL, metrics.recall,
        )

        results.append(GenerationCaseResult(
            case_id=case_id,
            dataset_name=dataset_name,
            question=question,
            model_answer=model_answer,
            reference_answer=reference,
            metrics=metrics,
            tags=tags,
        ))

    return results

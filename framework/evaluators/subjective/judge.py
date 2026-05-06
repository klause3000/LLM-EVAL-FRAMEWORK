"""主观评测核心逻辑：调用 ollama 被测模型 → judge 打分 → 返回结构化结果。"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import yaml

from framework.model_manager.ollama_manager import chat as ollama_chat
from .judge_clients import JudgeScore, build_judge_client

logger = logging.getLogger(__name__)


@dataclass
class CaseResult:
    """单条 case 的完整评测结果。"""

    case_id: str
    dataset_name: str
    question: str
    model_answer: str
    reference_answer: str
    dimension_scores: dict[str, float]
    overall_score: float
    judge_reasoning: str
    is_badcase: bool
    tags: list[str] = field(default_factory=list)


def run_subjective_eval(
    model_name: str,
    cases: list[dict],
    dataset_name: str,
    judge_config_path: str = "config/judge_config.yaml",
) -> list[CaseResult]:
    """
    对指定模型在给定 cases 上运行主观评测。

    :param model_name: ollama 模型名，如 "qwen2.5:7b"
    :param cases: 数据集中的 case 列表
    :param dataset_name: 数据集名称（用于入库标识）
    :param judge_config_path: judge 配置文件路径
    :return: 每条 case 的评测结果列表
    """
    with open(judge_config_path, "r", encoding="utf-8") as f:
        judge_cfg = yaml.safe_load(f)

    dimensions: list[dict] = judge_cfg.get("scoring_dimensions", [])
    badcase_threshold: float = float(judge_cfg.get("badcase_threshold", 6))

    judge_client = build_judge_client(judge_config_path)
    results: list[CaseResult] = []

    for case in cases:
        case_id: str = case["id"]
        question: str = case["question"]
        reference_answer: str = case.get("reference_answer") or ""
        tags: list[str] = case.get("tags") or []

        logger.info("[%s] 评测 case: %s", model_name, case_id)

        # 1. 获取被测模型的回答
        try:
            model_answer = ollama_chat(model_name, question)
        except Exception as e:
            logger.error("获取模型回答失败 case=%s: %s", case_id, e)
            model_answer = ""

        # 2. judge 打分
        try:
            score: JudgeScore = judge_client.score(
                question=question,
                model_answer=model_answer,
                reference_answer=reference_answer,
                dimensions=dimensions,
            )
        except Exception as e:
            logger.error("judge 评分失败 case=%s: %s", case_id, e)
            score = JudgeScore(
                dimension_scores={d["name"]: 0.0 for d in dimensions},
                overall_score=0.0,
                reasoning=f"评分异常: {e}",
            )

        # 3. badcase 判断：任意维度低于阈值即标记
        is_badcase = any(v < badcase_threshold for v in score.dimension_scores.values())

        results.append(
            CaseResult(
                case_id=case_id,
                dataset_name=dataset_name,
                question=question,
                model_answer=model_answer,
                reference_answer=reference_answer,
                dimension_scores=score.dimension_scores,
                overall_score=score.overall_score,
                judge_reasoning=score.reasoning,
                is_badcase=is_badcase,
                tags=tags,
            )
        )

    return results

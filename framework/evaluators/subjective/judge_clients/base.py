"""Judge client 抽象基类。"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class JudgeScore:
    dimension_scores: dict[str, float]   # {"生成质量": 8.0, "逻辑": 7.0, ...}
    overall_score: float                  # 各维度均值
    reasoning: str                        # 评分理由


class BaseJudgeClient(ABC):
    @abstractmethod
    def score(
        self,
        question: str,
        model_answer: str,
        reference_answer: str,
        dimensions: list[dict],
    ) -> JudgeScore:
        """
        对模型回答进行多维度评分。

        :param question: 原始问题
        :param model_answer: 被评测模型的回答
        :param reference_answer: 参考答案（可为空字符串）
        :param dimensions: 评分维度列表，每项 {"name": str, "description": str}
        :return: JudgeScore
        """

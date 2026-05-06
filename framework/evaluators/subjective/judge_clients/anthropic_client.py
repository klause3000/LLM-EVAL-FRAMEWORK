"""Anthropic Claude judge client。"""

import logging
import os

from anthropic import Anthropic

from .base import BaseJudgeClient, JudgeScore
from ._prompt import SYSTEM_PROMPT, build_judge_prompt
from ._parse import extract_json

logger = logging.getLogger(__name__)


class AnthropicJudgeClient(BaseJudgeClient):
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        max_retries: int = 3,
        timeout: int = 60,
    ):
        self.model = model
        self.client = Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
            max_retries=max_retries,
        )
        self.timeout = timeout

    def score(
        self,
        question: str,
        model_answer: str,
        reference_answer: str,
        dimensions: list[dict],
    ) -> JudgeScore:
        prompt = build_judge_prompt(question, model_answer, reference_answer, dimensions)

        message = self.client.messages.create(
            model=self.model,
            max_tokens=512,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
            timeout=self.timeout,
        )
        raw = message.content[0].text
        logger.debug("Anthropic judge 原始输出: %s", raw[:300])

        data = extract_json(raw)
        scores: dict[str, float] = {k: float(v) for k, v in data["scores"].items()}
        reasoning: str = data.get("reasoning", "")
        overall = sum(scores.values()) / len(scores) if scores else 0.0

        return JudgeScore(
            dimension_scores=scores,
            overall_score=round(overall, 2),
            reasoning=reasoning,
        )

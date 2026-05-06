"""OpenAI-compatible judge client（支持 OpenAI / 任意兼容接口）。"""

import logging
import os

from openai import OpenAI

from .base import BaseJudgeClient, JudgeScore
from ._prompt import SYSTEM_PROMPT, build_judge_prompt
from ._parse import extract_json

logger = logging.getLogger(__name__)


class OpenAIJudgeClient(BaseJudgeClient):
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        max_retries: int = 3,
        timeout: int = 60,
    ):
        self.model = model
        self.client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url,
            max_retries=max_retries,
            timeout=timeout,
        )

    def score(
        self,
        question: str,
        model_answer: str,
        reference_answer: str,
        dimensions: list[dict],
    ) -> JudgeScore:
        prompt = build_judge_prompt(question, model_answer, reference_answer, dimensions)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            # 仅在标准 OpenAI 端点启用 json_object 模式；兼容端点不一定支持
            # response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or ""
        logger.debug("OpenAI judge 原始输出: %s", raw[:300])

        data = extract_json(raw)
        scores: dict[str, float] = {k: float(v) for k, v in data["scores"].items()}
        reasoning: str = data.get("reasoning", "")
        overall = sum(scores.values()) / len(scores) if scores else 0.0

        return JudgeScore(
            dimension_scores=scores,
            overall_score=round(overall, 2),
            reasoning=reasoning,
        )

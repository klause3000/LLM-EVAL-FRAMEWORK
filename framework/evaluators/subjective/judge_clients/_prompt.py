"""Judge prompt 模板（供各 client 共用）。"""

SYSTEM_PROMPT = """你是一个专业的 AI 模型评测专家。
你的任务是对 AI 模型的回答进行多维度评分（每项 1-10 分），并给出简短的评分理由。
请严格按照要求的 JSON 格式输出，不要包含任何额外内容或 markdown 代码块。"""


def build_judge_prompt(
    question: str,
    model_answer: str,
    reference_answer: str,
    dimensions: list[dict],
) -> str:
    dims_lines = "\n".join(
        f'- {d["name"]}（1-10分）：{d["description"]}' for d in dimensions
    )
    score_keys = ", ".join(f'"{d["name"]}": <整数>' for d in dimensions)

    ref_block = f"\n【参考答案】\n{reference_answer}\n" if reference_answer.strip() else ""

    return f"""请对以下 AI 模型的回答进行评分。

【问题】
{question}
{ref_block}
【模型回答】
{model_answer}

【评分维度】
{dims_lines}

请按以下 JSON 格式输出（不要加 markdown 代码块，不要有多余文字）：
{{
  "scores": {{{score_keys}}},
  "reasoning": "整体评分理由，100字以内"
}}"""

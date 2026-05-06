"""AI 优化建议生成：调用 judge client 对评测结果生成文字分析。"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def _call_chat(judge_client, prompt: str, max_tokens: int = 600) -> str:
    """通用文本调用，兼容 OpenAI-compatible 和 Anthropic client。"""
    # OpenAIJudgeClient / OllamaJudgeClient 都有 .client (OpenAI SDK 实例)
    if hasattr(judge_client, "client"):
        resp = judge_client.client.chat.completions.create(
            model=judge_client.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""

    # AnthropicJudgeClient 有 .client (Anthropic SDK 实例)
    if hasattr(judge_client, "client") is False:
        try:
            resp = judge_client.client.messages.create(
                model=judge_client.model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text
        except Exception:
            pass

    raise AttributeError("无法识别的 judge client 类型")


def generate_single_run_advice(
    judge_client,
    model_name: str,
    total_cases: int,
    badcase_rate: float,
    avg_score: float | None,
    dimension_avg: dict[str, float],
    badcase_examples: list[dict],  # [{"case_id","question","judge_reasoning","tags"}]
) -> str:
    """生成单次评测优化建议，失败时返回空字符串。"""
    try:
        dim_lines = "\n".join(
            f"  {dim}: {score:.1f}/10" for dim, score in sorted(dimension_avg.items(), key=lambda x: x[1])
        )
        bc_lines = ""
        for bc in badcase_examples[:3]:
            tags = "、".join(bc.get("tags") or []) or "无标签"
            bc_lines += (
                f"\n  Case {bc['case_id']}（标签: {tags}）\n"
                f"  问题: {str(bc['question'])[:150]}\n"
                f"  评判理由: {str(bc.get('judge_reasoning', ''))[:150]}\n"
            )

        prompt = f"""你是一个 LLM 评测分析专家。请根据以下评测数据，给出 3~5 条具体的模型优化建议。
重点关注得分最低的维度和出现频率最高的问题模式。建议应具体可操作，不要泛泛而谈。

【评测概览】
模型: {model_name}
总 case 数: {total_cases}，badcase 率: {badcase_rate:.0%}，平均分: {f"{avg_score:.2f}/10" if avg_score else "N/A"}

【各维度平均分（从低到高）】
{dim_lines}

【代表性 Badcase】{bc_lines if bc_lines else "  无 badcase"}

【输出要求】
- 每条建议独占一行，以"• "开头
- 总字数不超过 400 字
- 聚焦最需要改进的方向，结合具体维度或 case 类型"""

        return _call_chat(judge_client, prompt)
    except Exception as e:
        logger.warning("AI 建议生成失败: %s", e)
        return ""


def generate_compare_advice(
    judge_client,
    model_name: str,
    run_labels: list[str],
    run_avg_scores: list[float | None],
    run_badcase_rates: list[float],
    delta_items: list[dict],  # [{"dataset","delta","trend"}]
    badcase_examples: list[dict],
) -> str:
    """生成多版本对比优化建议，失败时返回空字符串。"""
    try:
        version_lines = "\n".join(
            f"  {label}: 均分 {f'{score:.2f}' if score else 'N/A'}/10，badcase率 {rate:.0%}"
            for label, score, rate in zip(run_labels, run_avg_scores, run_badcase_rates)
        )

        delta_lines = ""
        for d in delta_items:
            sign = "▲" if d["trend"] == "up" else ("▼" if d["trend"] == "down" else "—")
            pct = f"{d['delta']*100:+.1f}%"
            delta_lines += f"  {d['dataset']}: {sign} {pct}\n"

        bc_lines = ""
        for bc in badcase_examples[:3]:
            tags = "、".join(bc.get("tags") or []) or "无标签"
            bc_lines += (
                f"\n  Case {bc['case_id']}（标签: {tags}）\n"
                f"  问题: {str(bc['question'])[:150]}\n"
                f"  评判理由: {str(bc.get('judge_reasoning', ''))[:150]}\n"
            )

        prompt = f"""你是一个 LLM 评测分析专家。请分析以下版本迭代数据，给出优化结论和建议。

【版本对比】
{version_lines}

【客观评测变化】
{delta_lines if delta_lines else "  无客观评测数据"}

【最新版本代表性 Badcase】{bc_lines if bc_lines else "  无 badcase"}

【输出要求】
- 第一行给出一句总结（整体提升/持平/退化）
- 随后列出 3~5 条具体建议，每条以"• "开头
- 总字数不超过 500 字
- 如有退化项，必须指出并分析可能原因"""

        return _call_chat(judge_client, prompt)
    except Exception as e:
        logger.warning("AI 对比建议生成失败: %s", e)
        return ""

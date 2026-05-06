"""BLEU / ROUGE / 召回率 计算，同时支持中文（字符级）和英文（词级）。"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class GenerationMetrics:
    bleu: float    # 0~100，sacrebleu corpus_bleu
    rouge1: float  # 0~1，F1
    rouge2: float  # 0~1，F1
    rougeL: float  # 0~1，F1
    recall: float  # 0~1，token-level recall（= ROUGE-1 recall）


def _is_chinese(text: str) -> bool:
    return bool(re.search(r"[一-鿿]", text))


def _tokenize(text: str) -> list[str]:
    """中文按字符切分，英文按空格切分（均小写）。"""
    if _is_chinese(text):
        return list(text.strip())
    return text.strip().lower().split()


def compute_metrics(hypothesis: str, reference: str) -> GenerationMetrics:
    """
    计算单条 case 的生成质量指标。

    :param hypothesis: 模型生成文本
    :param reference:  参考答案
    :return: GenerationMetrics
    """
    hyp = (hypothesis or "").strip()
    ref = (reference or "").strip()

    if not ref:
        return GenerationMetrics(bleu=0.0, rouge1=0.0, rouge2=0.0, rougeL=0.0, recall=0.0)

    # ── BLEU ──────────────────────────────────────────────────────────
    bleu_score = _compute_bleu(hyp, ref)

    # ── ROUGE ─────────────────────────────────────────────────────────
    r1, r2, rL, recall = _compute_rouge(hyp, ref)

    return GenerationMetrics(
        bleu=round(bleu_score, 2),
        rouge1=round(r1, 4),
        rouge2=round(r2, 4),
        rougeL=round(rL, 4),
        recall=round(recall, 4),
    )


def _compute_bleu(hyp: str, ref: str) -> float:
    try:
        from sacrebleu.metrics import BLEU

        use_char = _is_chinese(ref)
        tokenize = "char" if use_char else "13a"
        bleu = BLEU(tokenize=tokenize, smooth_method="exp")
        result = bleu.corpus_score([hyp], [[ref]])
        return float(result.score)
    except Exception:
        return 0.0


def _compute_rouge(hyp: str, ref: str) -> tuple[float, float, float, float]:
    """返回 (rouge1_f, rouge2_f, rougeL_f, rouge1_recall)。"""
    try:
        from rouge_score import rouge_scorer

        use_char = _is_chinese(ref)

        if use_char:
            # 字符级：把文本转成空格分隔的字符序列，再交给 rouge_scorer
            hyp_spaced = " ".join(list(hyp))
            ref_spaced = " ".join(list(ref))
        else:
            hyp_spaced = hyp
            ref_spaced = ref

        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=not use_char)
        scores = scorer.score(ref_spaced, hyp_spaced)

        rouge1_recall = scores["rouge1"].recall
        return (
            scores["rouge1"].fmeasure,
            scores["rouge2"].fmeasure,
            scores["rougeL"].fmeasure,
            rouge1_recall,
        )
    except Exception:
        return 0.0, 0.0, 0.0, 0.0

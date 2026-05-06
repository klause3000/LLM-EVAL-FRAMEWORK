"""
评测报告生成：控制台打印版本对比表 + badcase 覆盖分析。
输出使用纯文本表格，无需额外依赖。
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session

from framework.storage.db import (
    get_badcase_summary,
    get_latest_runs,
    get_objective_comparison,
    get_subjective_comparison,
)

logger = logging.getLogger(__name__)

_SEP = "-" * 80


def _fmt_score(score: Optional[float], as_pct: bool = False) -> str:
    if score is None:
        return "N/A"
    if as_pct:
        return f"{score * 100:.1f}%"
    return f"{score:.2f}"


def _diff_arrow(prev: Optional[float], curr: Optional[float]) -> str:
    if prev is None or curr is None:
        return ""
    delta = curr - prev
    if delta > 0:
        return f"  (+{delta:.2f})"
    if delta < 0:
        return f"  ({delta:.2f})"
    return "  (=)"


# ---------------------------------------------------------------------------
# 客观评测对比
# ---------------------------------------------------------------------------

def print_objective_comparison(session: Session, model_name: str, last_n: int = 2) -> None:
    """打印最近 last_n 次客观评测结果对比表。"""
    runs = get_latest_runs(session, model_name, "objective", n=last_n)
    if not runs:
        print(f"\n[客观评测] 模型 {model_name} 暂无历史记录\n")
        return

    run_ids = [r.id for r in runs]
    data = get_objective_comparison(session, model_name, run_ids=run_ids)

    # 按时间倒序展示（最新在右）
    runs_sorted = sorted(runs, key=lambda r: r.created_at)
    headers = ["数据集"] + [
        f"Run#{r.id}\n{r.created_at.strftime('%m-%d %H:%M')}\n{r.model_version or ''}"
        for r in runs_sorted
    ]

    print(f"\n{'=' * 80}")
    print(f"  客观评测版本对比  |  模型: {model_name}")
    print(_SEP)

    col_w = 18
    header_line = f"{'数据集':<20}" + "".join(f"{h:<{col_w}}" for h in headers[1:])
    print(header_line)
    print(_SEP)

    for item in data:
        ds = item["dataset"]
        score_by_run: dict[int, Optional[float]] = {
            s["run_id"]: s["score"] for s in item["scores"]
        }
        row = f"{ds:<20}"
        prev_score = None
        for run in runs_sorted:
            score = score_by_run.get(run.id)
            cell = _fmt_score(score, as_pct=True) + _diff_arrow(prev_score, score)
            row += f"{cell:<{col_w}}"
            prev_score = score
        print(row)

    print(_SEP + "\n")


# ---------------------------------------------------------------------------
# 主观评测对比 + badcase 分析
# ---------------------------------------------------------------------------

def print_subjective_comparison(session: Session, model_name: str, last_n: int = 2) -> None:
    """打印主观评测对比表，并展示 badcase 覆盖情况。"""
    runs = get_latest_runs(session, model_name, "subjective", n=last_n)
    if not runs:
        print(f"\n[主观评测] 模型 {model_name} 暂无历史记录\n")
        return

    run_ids = [r.id for r in runs]
    runs_sorted = sorted(runs, key=lambda r: r.created_at)

    # 1. 汇总得分对比（per run）
    print(f"\n{'=' * 80}")
    print(f"  主观评测版本对比  |  模型: {model_name}")
    print(_SEP)

    for run in runs_sorted:
        summary = get_badcase_summary(session, run.id)
        ts = run.created_at.strftime("%Y-%m-%d %H:%M")
        print(
            f"  Run#{run.id}  {ts}  v={run.model_version or '-'}"
            f"  |  总 case: {summary['total_cases']}"
            f"  |  badcase: {summary['badcase_count']} ({summary['badcase_rate']*100:.0f}%)"
            f"  |  均分: {_fmt_score(summary['avg_overall_score'])}"
        )

    print(_SEP)

    # 2. Case 级别 badcase 覆盖对比
    case_data = get_subjective_comparison(session, model_name, run_ids=run_ids)
    badcase_items = [c for c in case_data if any(r["is_badcase"] for r in c["results"])]

    if badcase_items:
        print(f"\n  Badcase 覆盖情况（共 {len(badcase_items)} 个问题曾被标记为 badcase）\n")
        col_w = 12
        header = f"  {'Case ID':<20}{'数据集':<16}" + "".join(
            f"Run#{r.id:<{col_w-4}}" for r in runs_sorted
        )
        print(header)
        print("  " + "-" * (36 + col_w * len(runs_sorted)))

        score_by_run: dict[int, dict]
        for item in badcase_items:
            score_by_run = {r["run_id"]: r for r in item["results"]}
            row = f"  {item['case_id']:<20}{item['dataset']:<16}"
            for run in runs_sorted:
                r = score_by_run.get(run.id)
                if r is None:
                    cell = "  -"
                else:
                    flag = "BAD" if r["is_badcase"] else "OK "
                    cell = f"  {flag}({_fmt_score(r['overall_score'])})"
                row += f"{cell:<{col_w}}"
            print(row)
    else:
        print("\n  无 badcase 记录。\n")

    print(_SEP + "\n")


# ---------------------------------------------------------------------------
# 入口：一键打印全部对比
# ---------------------------------------------------------------------------

def print_full_report(session: Session, model_name: str, last_n: int = 2) -> None:
    print_objective_comparison(session, model_name, last_n=last_n)
    print_subjective_comparison(session, model_name, last_n=last_n)

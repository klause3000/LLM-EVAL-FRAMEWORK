"""
数据库操作：初始化、写入、查询（版本对比 / badcase 覆盖分析）。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session, sessionmaker

from .models import Base, EvalRun, ObjectiveResult, SubjectiveResult, GenerationResult, SafetyResult

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "eval_results.db"
DEFAULT_DB_URL = f"sqlite:///{DEFAULT_DB_PATH}"

_engine = None
_SessionFactory = None


def init_db(db_url: str = DEFAULT_DB_URL):
    """创建表结构（幂等），返回 engine。"""
    global _engine, _SessionFactory
    _engine = create_engine(db_url, echo=False, future=True)
    Base.metadata.create_all(_engine)
    _SessionFactory = sessionmaker(bind=_engine, expire_on_commit=False)
    logger.info("数据库已初始化: %s", db_url)
    return _engine


def get_session() -> Session:
    """返回新 Session；调用方负责 commit/close。"""
    if _SessionFactory is None:
        init_db()
    return _SessionFactory()


# ---------------------------------------------------------------------------
# 写入
# ---------------------------------------------------------------------------

def save_objective_results(
    session: Session,
    model_name: str,
    model_version: str,
    results: dict[str, dict],
) -> EvalRun:
    """
    将客观评测结果写入数据库。
    results 格式: {dataset_name: {"score": float, "num_samples": int, "raw": dict, "error": str|None}}
    """
    run = EvalRun(model_name=model_name, model_version=model_version, run_type="objective")
    session.add(run)
    session.flush()  # 获取 run.id

    for dataset_name, res in results.items():
        obj_result = ObjectiveResult(
            run_id=run.id,
            dataset_name=dataset_name,
            score=res.get("score"),
            num_samples=res.get("num_samples", 0),
            raw_metrics=res.get("raw"),
            error=res.get("error"),
        )
        session.add(obj_result)

    session.commit()
    logger.info("客观评测结果已入库: run_id=%d model=%s", run.id, model_name)
    return run


def save_subjective_results(
    session: Session,
    model_name: str,
    model_version: str,
    case_results: list,  # list[CaseResult] from judge.py
) -> EvalRun:
    """
    将主观评测结果写入数据库。
    case_results: list[CaseResult]（避免循环导入，使用 duck-typing）
    """
    run = EvalRun(model_name=model_name, model_version=model_version, run_type="subjective")
    session.add(run)
    session.flush()

    for cr in case_results:
        subj_result = SubjectiveResult(
            run_id=run.id,
            dataset_name=cr.dataset_name,
            case_id=cr.case_id,
            question=cr.question,
            model_answer=cr.model_answer,
            reference_answer=cr.reference_answer,
            dimension_scores=cr.dimension_scores,
            overall_score=cr.overall_score,
            judge_reasoning=cr.judge_reasoning,
            is_badcase=cr.is_badcase,
            tags=cr.tags,
        )
        session.add(subj_result)

    session.commit()
    logger.info("主观评测结果已入库: run_id=%d model=%s cases=%d", run.id, model_name, len(case_results))
    return run


# ---------------------------------------------------------------------------
# 查询
# ---------------------------------------------------------------------------

def get_latest_runs(session: Session, model_name: str, run_type: str, n: int = 2) -> list[EvalRun]:
    """获取指定模型最近 n 次评测运行。"""
    stmt = (
        select(EvalRun)
        .where(EvalRun.model_name == model_name, EvalRun.run_type == run_type)
        .order_by(EvalRun.created_at.desc())
        .limit(n)
    )
    return list(session.scalars(stmt))


def get_objective_comparison(
    session: Session, model_name: str, run_ids: Optional[list[int]] = None
) -> list[dict]:
    """
    返回客观评测各 dataset 的历史得分列表，用于版本对比。
    格式: [{"dataset": str, "scores": [{"run_id": int, "score": float, "created_at": ...}]}]
    """
    stmt = (
        select(ObjectiveResult, EvalRun.created_at, EvalRun.model_version)
        .join(EvalRun, ObjectiveResult.run_id == EvalRun.id)
        .where(EvalRun.model_name == model_name)
    )
    if run_ids:
        stmt = stmt.where(ObjectiveResult.run_id.in_(run_ids))
    stmt = stmt.order_by(EvalRun.created_at.asc())

    rows = session.execute(stmt).all()

    # 按 dataset 聚合
    by_dataset: dict[str, list] = {}
    for obj_res, created_at, model_version in rows:
        ds = obj_res.dataset_name
        by_dataset.setdefault(ds, []).append(
            {
                "run_id": obj_res.run_id,
                "score": obj_res.score,
                "num_samples": obj_res.num_samples,
                "model_version": model_version,
                "created_at": created_at,
            }
        )

    return [{"dataset": ds, "scores": scores} for ds, scores in by_dataset.items()]


def get_subjective_comparison(
    session: Session, model_name: str, run_ids: Optional[list[int]] = None
) -> list[dict]:
    """
    返回主观评测 per-case 跨版本得分，用于 badcase 覆盖分析。
    格式: [{"case_id": str, "dataset": str, "results": [{"run_id": int, "overall_score": float, "is_badcase": bool}]}]
    """
    stmt = (
        select(SubjectiveResult, EvalRun.created_at, EvalRun.model_version)
        .join(EvalRun, SubjectiveResult.run_id == EvalRun.id)
        .where(EvalRun.model_name == model_name)
    )
    if run_ids:
        stmt = stmt.where(SubjectiveResult.run_id.in_(run_ids))
    stmt = stmt.order_by(EvalRun.created_at.asc())

    rows = session.execute(stmt).all()

    by_case: dict[str, dict] = {}
    for subj_res, created_at, model_version in rows:
        key = f"{subj_res.dataset_name}::{subj_res.case_id}"
        if key not in by_case:
            by_case[key] = {
                "case_id": subj_res.case_id,
                "dataset": subj_res.dataset_name,
                "question": subj_res.question,
                "results": [],
            }
        by_case[key]["results"].append(
            {
                "run_id": subj_res.run_id,
                "overall_score": subj_res.overall_score,
                "dimension_scores": subj_res.dimension_scores,
                "is_badcase": subj_res.is_badcase,
                "model_version": model_version,
                "created_at": created_at,
            }
        )

    return list(by_case.values())


def save_generation_results(
    session: Session,
    model_name: str,
    model_version: str,
    case_results: list,  # list[GenerationCaseResult]
) -> EvalRun:
    """将生成质量评测结果写入数据库。"""
    run = EvalRun(model_name=model_name, model_version=model_version, run_type="generation")
    session.add(run)
    session.flush()

    for cr in case_results:
        m = cr.metrics
        session.add(GenerationResult(
            run_id=run.id,
            dataset_name=cr.dataset_name,
            case_id=cr.case_id,
            question=cr.question,
            model_answer=cr.model_answer,
            reference_answer=cr.reference_answer,
            bleu=m.bleu,
            rouge1=m.rouge1,
            rouge2=m.rouge2,
            rougeL=m.rougeL,
            recall=m.recall,
            tags=cr.tags,
        ))

    session.commit()
    logger.info("生成质量结果已入库: run_id=%d model=%s cases=%d", run.id, model_name, len(case_results))
    return run


def get_generation_summary(session: Session, run_id: int) -> dict:
    """统计某次生成质量评测的平均指标。"""
    rows = session.scalars(
        select(GenerationResult).where(GenerationResult.run_id == run_id)
    ).all()
    if not rows:
        return {"run_id": run_id, "total_cases": 0}

    def _avg(vals):
        vals = [v for v in vals if v is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    return {
        "run_id": run_id,
        "total_cases": len(rows),
        "avg_bleu": _avg([r.bleu for r in rows]),
        "avg_rouge1": _avg([r.rouge1 for r in rows]),
        "avg_rouge2": _avg([r.rouge2 for r in rows]),
        "avg_rougeL": _avg([r.rougeL for r in rows]),
        "avg_recall": _avg([r.recall for r in rows]),
    }


def get_badcase_summary(session: Session, run_id: int) -> dict:
    """统计某次运行的 badcase 情况。"""
    total = session.scalar(
        select(func.count()).where(SubjectiveResult.run_id == run_id)
    ) or 0
    bad = session.scalar(
        select(func.count()).where(
            SubjectiveResult.run_id == run_id,
            SubjectiveResult.is_badcase == True,  # noqa: E712
        )
    ) or 0
    avg_score = session.scalar(
        select(func.avg(SubjectiveResult.overall_score)).where(
            SubjectiveResult.run_id == run_id
        )
    )
    return {
        "run_id": run_id,
        "total_cases": total,
        "badcase_count": bad,
        "badcase_rate": round(bad / total, 3) if total else 0,
        "avg_overall_score": round(avg_score, 2) if avg_score else None,
    }


def save_safety_results(
    session: Session,
    model_name: str,
    model_version: str,
    case_results: list,  # list[SafetyCaseResult]
) -> EvalRun:
    """将安全性评测结果写入数据库。"""
    run = EvalRun(model_name=model_name, model_version=model_version, run_type="safety")
    session.add(run)
    session.flush()

    for cr in case_results:
        session.add(SafetyResult(
            run_id=run.id,
            dataset_name=cr.dataset_name,
            case_id=cr.case_id,
            category=cr.category,
            question=cr.question,
            model_answer=cr.model_answer,
            is_safe=cr.is_safe,
            judge_reasoning=cr.judge_reasoning,
            tags=cr.tags,
        ))

    session.commit()
    logger.info("安全评测结果已入库: run_id=%d model=%s cases=%d", run.id, model_name, len(case_results))
    return run


def get_safety_summary(session: Session, run_id: int) -> dict:
    """统计某次安全评测的整体安全率及分类安全率。"""
    rows = session.scalars(
        select(SafetyResult).where(SafetyResult.run_id == run_id)
    ).all()
    if not rows:
        return {"run_id": run_id, "total_cases": 0}

    total = len(rows)
    safe_count = sum(1 for r in rows if r.is_safe)
    unsafe_count = total - safe_count

    # 按分类统计
    by_category: dict[str, dict] = {}
    for r in rows:
        cat = r.category or "general"
        by_category.setdefault(cat, {"total": 0, "safe": 0})
        by_category[cat]["total"] += 1
        if r.is_safe:
            by_category[cat]["safe"] += 1

    category_rates = {
        cat: round(v["safe"] / v["total"], 3) if v["total"] else None
        for cat, v in by_category.items()
    }

    return {
        "run_id": run_id,
        "total_cases": total,
        "safe_count": safe_count,
        "unsafe_count": unsafe_count,
        "safety_rate": round(safe_count / total, 3) if total else None,
        "by_category": category_rates,
    }

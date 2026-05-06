"""HTML 评测报告生成器：单次报告 + 多版本对比报告 + 会话聚合报告。"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import plotly.graph_objects as go
from jinja2 import Environment, FileSystemLoader
from sqlalchemy import select
from sqlalchemy.orm import Session

from framework.storage.models import EvalRun, ObjectiveResult, SubjectiveResult, GenerationResult, SafetyResult
from framework.storage.db import get_latest_runs, get_generation_summary

logger = logging.getLogger(__name__)

TEMPLATES_DIR = Path(__file__).parent / "templates"

# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class RunInfo:
    run_id: int
    model_name: str
    model_version: Optional[str]
    run_type: str
    created_at: str
    label: str  # "Run#5 · v1.2 · 2026-04-19 14:30"


@dataclass
class ObjDataset:
    name: str
    score: Optional[float]   # 0~1
    num_samples: int
    error: Optional[str]


@dataclass
class SubjSummary:
    total_cases: int
    badcase_count: int
    badcase_rate: float
    avg_score: Optional[float]
    scores: list[float]                    # 所有 case overall_score，用于 histogram
    dimension_avg: dict[str, float]        # {维度名: 平均分}
    tag_dim_avg: dict[str, dict[str, float]]  # {tag: {维度名: 平均分}}


@dataclass
class BadcaseItem:
    case_id: str
    dataset_name: str
    tags: list[str]
    question: str
    model_answer: str
    reference_answer: str
    overall_score: float
    dimension_scores: dict[str, float]
    judge_reasoning: str


@dataclass
class GenSummary:
    total_cases: int
    avg_bleu: Optional[float]
    avg_rouge1: Optional[float]
    avg_rouge2: Optional[float]
    avg_rougeL: Optional[float]
    avg_recall: Optional[float]
    case_metrics: list[dict]  # [{"case_id", "bleu", "rouge1", "rouge2", "rougeL", "recall", "tags"}]


@dataclass
class SingleRunData:
    run: RunInfo
    obj_datasets: list[ObjDataset]
    subj: Optional[SubjSummary]
    badcases: list[BadcaseItem]
    gen: Optional[GenSummary]
    ai_advice: str


@dataclass
class CompareRunData:
    run: RunInfo
    obj_datasets: list[ObjDataset]
    subj: Optional[SubjSummary]
    badcases: list[BadcaseItem]
    gen: Optional[GenSummary]


@dataclass
class DeltaInfo:
    dataset: str
    old_score: Optional[float]
    new_score: Optional[float]
    delta: float
    trend: str   # "up" | "down" | "flat"
    pct_str: str  # "+5.2%" / "-1.1%" / "±0.0%"


@dataclass
class CompareData:
    model_name: str
    runs: list[CompareRunData]   # 从旧到新
    obj_deltas: list[DeltaInfo]
    ai_advice: str


@dataclass
class DatasetSubjData:
    dataset_name: str
    run_id: int
    total_cases: int
    badcase_count: int
    badcase_rate: float
    avg_score: Optional[float]


@dataclass
class DatasetGenData:
    dataset_name: str
    run_id: int
    total_cases: int
    avg_bleu: Optional[float]
    avg_rouge1: Optional[float]
    avg_rouge2: Optional[float]
    avg_rougeL: Optional[float]
    avg_recall: Optional[float]


@dataclass
class DatasetSafetyData:
    dataset_name: str
    run_id: int
    total_cases: int
    safe_count: int
    unsafe_count: int
    safety_rate: Optional[float]
    by_category: dict[str, Optional[float]]  # {category: safety_rate}


@dataclass
class SafetyUnsafeItem:
    case_id: str
    dataset_name: str
    category: str
    question: str
    model_answer: str
    judge_reasoning: str
    tags: list[str]


@dataclass
class ModelSessionData:
    model_name: str
    obj_run_id: Optional[int]
    obj_datasets: list       # list[ObjDataset]
    subj_datasets: list      # list[DatasetSubjData]
    combined_subj: Optional[SubjSummary]
    badcases: list           # list[BadcaseItem]
    gen_datasets: list       # list[DatasetGenData]
    safety_datasets: list    # list[DatasetSafetyData]
    unsafe_cases: list       # list[SafetyUnsafeItem]


@dataclass
class SessionData:
    session_ts: str
    total_runs: int
    models: list             # list[ModelSessionData]


# ---------------------------------------------------------------------------
# 数据查询 & 聚合
# ---------------------------------------------------------------------------

def _make_run_info(run: EvalRun) -> RunInfo:
    ver = run.model_version or "-"
    ts = run.created_at.strftime("%Y-%m-%d %H:%M") if run.created_at else "-"
    label = f"Run#{run.id} · v{ver} · {ts}"
    return RunInfo(
        run_id=run.id,
        model_name=run.model_name,
        model_version=run.model_version,
        run_type=run.run_type,
        created_at=ts,
        label=label,
    )


def _query_obj_datasets(session: Session, run_id: int) -> list[ObjDataset]:
    rows = session.scalars(
        select(ObjectiveResult).where(ObjectiveResult.run_id == run_id)
    ).all()
    return [
        ObjDataset(
            name=r.dataset_name,
            score=r.score,
            num_samples=r.num_samples or 0,
            error=r.error,
        )
        for r in rows
    ]


def _query_subj_summary(session: Session, run_id: int) -> Optional[SubjSummary]:
    rows = session.scalars(
        select(SubjectiveResult).where(SubjectiveResult.run_id == run_id)
    ).all()
    if not rows:
        return None

    total = len(rows)
    badcase_count = sum(1 for r in rows if r.is_badcase)
    scores = [r.overall_score for r in rows if r.overall_score is not None]
    avg_score = round(sum(scores) / len(scores), 2) if scores else None

    # 各维度均值
    dim_buckets: dict[str, list[float]] = defaultdict(list)
    tag_dim_buckets: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for r in rows:
        for dim, val in (r.dimension_scores or {}).items():
            dim_buckets[dim].append(float(val))
        for tag in (r.tags or []):
            for dim, val in (r.dimension_scores or {}).items():
                tag_dim_buckets[tag][dim].append(float(val))

    dimension_avg = {dim: round(sum(v) / len(v), 2) for dim, v in dim_buckets.items()}

    # 能力标签聚合（过滤 case 数 < 1 的 tag）
    tag_dim_avg: dict[str, dict[str, float]] = {}
    for tag, dims in tag_dim_buckets.items():
        tag_dim_avg[tag] = {dim: round(sum(v) / len(v), 2) for dim, v in dims.items()}

    return SubjSummary(
        total_cases=total,
        badcase_count=badcase_count,
        badcase_rate=round(badcase_count / total, 3) if total else 0.0,
        avg_score=avg_score,
        scores=scores,
        dimension_avg=dimension_avg,
        tag_dim_avg=tag_dim_avg,
    )


def _query_badcases(session: Session, run_id: int) -> list[BadcaseItem]:
    rows = session.scalars(
        select(SubjectiveResult)
        .where(SubjectiveResult.run_id == run_id, SubjectiveResult.is_badcase == True)  # noqa: E712
        .order_by(SubjectiveResult.dataset_name, SubjectiveResult.case_id)
    ).all()
    return [
        BadcaseItem(
            case_id=r.case_id,
            dataset_name=r.dataset_name,
            tags=r.tags or [],
            question=r.question or "",
            model_answer=r.model_answer or "",
            reference_answer=r.reference_answer or "",
            overall_score=r.overall_score or 0.0,
            dimension_scores=r.dimension_scores or {},
            judge_reasoning=r.judge_reasoning or "",
        )
        for r in rows
    ]


def _query_gen_summary(session: Session, run_id: int) -> Optional[GenSummary]:
    rows = session.scalars(
        select(GenerationResult).where(GenerationResult.run_id == run_id)
    ).all()
    if not rows:
        return None

    def _avg(vals):
        vals = [v for v in vals if v is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    case_metrics = [
        {
            "case_id": r.case_id,
            "dataset_name": r.dataset_name,
            "tags": r.tags or [],
            "bleu": r.bleu,
            "rouge1": r.rouge1,
            "rouge2": r.rouge2,
            "rougeL": r.rougeL,
            "recall": r.recall,
            "question": (r.question or "")[:80],
        }
        for r in rows
    ]

    return GenSummary(
        total_cases=len(rows),
        avg_bleu=_avg([r.bleu for r in rows]),
        avg_rouge1=_avg([r.rouge1 for r in rows]),
        avg_rouge2=_avg([r.rouge2 for r in rows]),
        avg_rougeL=_avg([r.rougeL for r in rows]),
        avg_recall=_avg([r.recall for r in rows]),
        case_metrics=case_metrics,
    )


# ---------------------------------------------------------------------------
# Plotly 图表生成
# ---------------------------------------------------------------------------

# 颜色方案
_COLORS = ["#4f8ef7", "#f76c6c", "#43c98a", "#ffc107", "#9c68d4", "#17becf"]


def _embed_plotlyjs() -> str:
    """读取本地 plotly.min.js 并返回 <script> 标签（自包含，无需 CDN）。"""
    try:
        import plotly
        js_path = Path(plotly.__file__).parent / "package_data" / "plotly.min.js"
        if js_path.exists():
            js_content = js_path.read_text(encoding="utf-8")
            return f"<script>{js_content}</script>"
    except Exception as e:
        logger.warning("无法读取本地 plotly.min.js，回退到 CDN: %s", e)
    return '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>'


def _fig_html(fig: go.Figure, height: int = 400) -> str:
    """将 Figure 转为 HTML div 片段（不含 plotly.js，由模板统一引入）。"""
    fig.update_layout(
        height=height,
        margin=dict(l=40, r=40, t=40, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="system-ui, sans-serif", size=13),
    )
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True})


def make_obj_bar_chart(datasets: list[ObjDataset]) -> str:
    """客观得分柱状图。"""
    valid = [d for d in datasets if d.score is not None]
    if not valid:
        return "<p class='no-data'>无客观评测数据</p>"

    names = [d.name for d in valid]
    scores = [round(d.score * 100, 1) for d in valid]
    colors = [_COLORS[i % len(_COLORS)] for i in range(len(valid))]

    fig = go.Figure(go.Bar(
        x=names, y=scores,
        marker_color=colors,
        text=[f"{s}%" for s in scores],
        textposition="outside",
    ))
    fig.update_layout(
        title="客观评测得分",
        yaxis=dict(title="准确率 (%)", range=[0, 110]),
        showlegend=False,
    )
    return _fig_html(fig, height=350)


def make_score_histogram(scores: list[float], title: str = "主观评分分布") -> str:
    """主观评分分布直方图。"""
    if not scores:
        return "<p class='no-data'>无主观评测数据</p>"

    fig = go.Figure(go.Histogram(
        x=scores,
        xbins=dict(start=0, end=10, size=1),
        marker_color=_COLORS[0],
        opacity=0.8,
    ))
    fig.update_layout(
        title=title,
        xaxis=dict(title="得分", range=[0, 10]),
        yaxis=dict(title="case 数"),
        bargap=0.1,
    )
    return _fig_html(fig, height=300)


def make_radar_chart(tag_dim_avg: dict[str, dict[str, float]], title: str = "能力维度雷达图") -> str:
    """能力标签雷达图：每个 tag 一条线，各评分维度为雷达轴。"""
    if not tag_dim_avg:
        return "<p class='no-data'>无标签数据（需要 case 带 tags 字段）</p>"

    # 取所有维度名（取并集，保持顺序）
    all_dims: list[str] = []
    for dims in tag_dim_avg.values():
        for d in dims:
            if d not in all_dims:
                all_dims.append(d)

    if not all_dims:
        return "<p class='no-data'>无维度数据</p>"

    # tag 过多时只取 top-6（按 case 数量，这里按维度均分均值排序）
    tag_items = list(tag_dim_avg.items())
    if len(tag_items) > 6:
        tag_items = sorted(
            tag_items,
            key=lambda x: sum(x[1].values()) / len(x[1]) if x[1] else 0,
            reverse=True,
        )[:6]

    fig = go.Figure()
    for i, (tag, dims) in enumerate(tag_items):
        r_vals = [dims.get(d, 0) for d in all_dims]
        r_vals_closed = r_vals + [r_vals[0]]  # 闭合
        theta_closed = all_dims + [all_dims[0]]
        fig.add_trace(go.Scatterpolar(
            r=r_vals_closed,
            theta=theta_closed,
            fill="toself",
            name=tag,
            line_color=_COLORS[i % len(_COLORS)],
            opacity=0.7,
        ))

    fig.update_layout(
        title=title,
        polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
        showlegend=True,
    )
    return _fig_html(fig, height=420)


def make_compare_bar_chart(
    run_labels: list[str],
    datasets: list[str],
    scores_per_run: list[dict[str, Optional[float]]],  # [{dataset: score}, ...]
) -> str:
    """多版本客观分数分组柱状图。"""
    if not datasets or not scores_per_run:
        return "<p class='no-data'>无客观对比数据</p>"

    fig = go.Figure()
    for i, (label, scores) in enumerate(zip(run_labels, scores_per_run)):
        y_vals = [
            round(scores.get(ds, 0) * 100, 1) if scores.get(ds) is not None else 0
            for ds in datasets
        ]
        fig.add_trace(go.Bar(
            name=label,
            x=datasets,
            y=y_vals,
            marker_color=_COLORS[i % len(_COLORS)],
            text=[f"{v}%" for v in y_vals],
            textposition="outside",
        ))

    fig.update_layout(
        title="客观评测版本对比",
        barmode="group",
        yaxis=dict(title="准确率 (%)", range=[0, 120]),
    )
    return _fig_html(fig, height=380)


def make_subj_trend_chart(
    run_labels: list[str],
    avg_scores: list[Optional[float]],
    badcase_rates: list[float],
) -> str:
    """主观评分趋势折线图（均分 + badcase率双轴）。"""
    if not run_labels:
        return "<p class='no-data'>无主观对比数据</p>"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=run_labels,
        y=[s if s is not None else None for s in avg_scores],
        mode="lines+markers",
        name="平均分",
        line=dict(color=_COLORS[0], width=2),
        marker=dict(size=8),
        yaxis="y1",
    ))
    fig.add_trace(go.Scatter(
        x=run_labels,
        y=[r * 100 for r in badcase_rates],
        mode="lines+markers",
        name="badcase率 (%)",
        line=dict(color=_COLORS[1], width=2, dash="dot"),
        marker=dict(size=8),
        yaxis="y2",
    ))
    fig.update_layout(
        title="主观评测趋势",
        yaxis=dict(title="平均分 (0~10)", range=[0, 10]),
        yaxis2=dict(title="badcase率 (%)", overlaying="y", side="right", range=[0, 100]),
        legend=dict(x=0.01, y=0.99),
    )
    return _fig_html(fig, height=350)


def make_gen_metrics_chart(case_metrics: list[dict], title: str = "逐条生成指标") -> str:
    """生成质量逐条指标柱状图（每条 case 的 ROUGE-L 和召回率）。"""
    if not case_metrics:
        return "<p class='no-data'>无生成质量数据</p>"

    case_ids = [m["case_id"] for m in case_metrics]
    rougeL = [m.get("rougeL") or 0 for m in case_metrics]
    recall = [m.get("recall") or 0 for m in case_metrics]
    bleu = [(m.get("bleu") or 0) / 100 for m in case_metrics]  # 归一化到 0~1

    fig = go.Figure()
    fig.add_trace(go.Bar(name="ROUGE-L", x=case_ids, y=rougeL, marker_color=_COLORS[0]))
    fig.add_trace(go.Bar(name="召回率", x=case_ids, y=recall, marker_color=_COLORS[2]))
    fig.add_trace(go.Bar(name="BLEU/100", x=case_ids, y=bleu, marker_color=_COLORS[3]))
    fig.update_layout(
        title=title,
        barmode="group",
        yaxis=dict(title="得分 (0~1)", range=[0, 1.05]),
        xaxis=dict(tickangle=-30),
    )
    return _fig_html(fig, height=380)


def make_gen_compare_chart(
    run_labels: list[str],
    gen_summaries: list[Optional["GenSummary"]],
) -> str:
    """多版本生成指标对比折线图。"""
    if not run_labels:
        return "<p class='no-data'>无生成质量对比数据</p>"

    metrics_map = {
        "ROUGE-1": [g.avg_rouge1 if g else None for g in gen_summaries],
        "ROUGE-L": [g.avg_rougeL if g else None for g in gen_summaries],
        "召回率":  [g.avg_recall if g else None for g in gen_summaries],
    }

    fig = go.Figure()
    for i, (name, vals) in enumerate(metrics_map.items()):
        fig.add_trace(go.Scatter(
            x=run_labels, y=vals,
            mode="lines+markers",
            name=name,
            line=dict(color=_COLORS[i], width=2),
            marker=dict(size=8),
        ))
    fig.update_layout(
        title="生成质量指标趋势",
        yaxis=dict(title="得分 (0~1)", range=[0, 1.05]),
    )
    return _fig_html(fig, height=350)


# ---------------------------------------------------------------------------
# delta 计算
# ---------------------------------------------------------------------------

def _compute_deltas(
    old_datasets: list[ObjDataset],
    new_datasets: list[ObjDataset],
) -> list[DeltaInfo]:
    old_map = {d.name: d.score for d in old_datasets}
    new_map = {d.name: d.score for d in new_datasets}
    all_names = sorted(set(old_map) | set(new_map))

    deltas = []
    for name in all_names:
        old_s = old_map.get(name)
        new_s = new_map.get(name)
        if old_s is None or new_s is None:
            delta = 0.0
            trend = "flat"
            pct_str = "N/A"
        else:
            delta = new_s - old_s
            trend = "up" if delta > 0.005 else ("down" if delta < -0.005 else "flat")
            pct_str = f"{delta * 100:+.1f}%"
        deltas.append(DeltaInfo(
            dataset=name,
            old_score=old_s,
            new_score=new_s,
            delta=delta,
            trend=trend,
            pct_str=pct_str,
        ))
    return deltas


# ---------------------------------------------------------------------------
# 主入口：生成报告
# ---------------------------------------------------------------------------

def _render(template_name: str, **ctx) -> str:
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=False,
    )
    tmpl = env.get_template(template_name)
    return tmpl.render(**ctx)


def build_single_run_report(
    session: Session,
    run_id: int,
    output_path: str,
    judge_config_path: str = "config/judge_config.yaml",
    with_ai_advice: bool = True,
) -> None:
    """生成单次评测 HTML 报告，写入 output_path。"""
    run = session.get(EvalRun, run_id)
    if not run:
        raise ValueError(f"Run #{run_id} 不存在")

    run_info = _make_run_info(run)
    obj_datasets = _query_obj_datasets(session, run_id)
    subj = _query_subj_summary(session, run_id)
    badcases = _query_badcases(session, run_id)
    gen = _query_gen_summary(session, run_id)

    # AI 建议
    ai_advice = ""
    if with_ai_advice and subj:
        try:
            from framework.evaluators.subjective.judge_clients import build_judge_client
            from framework.reporters.ai_advice import generate_single_run_advice
            client = build_judge_client(judge_config_path)
            bc_examples = [
                {
                    "case_id": bc.case_id,
                    "question": bc.question,
                    "judge_reasoning": bc.judge_reasoning,
                    "tags": bc.tags,
                }
                for bc in badcases
            ]
            ai_advice = generate_single_run_advice(
                client,
                model_name=run.model_name,
                total_cases=subj.total_cases,
                badcase_rate=subj.badcase_rate,
                avg_score=subj.avg_score,
                dimension_avg=subj.dimension_avg,
                badcase_examples=bc_examples,
            )
        except Exception as e:
            logger.warning("AI 建议生成失败（不影响报告）: %s", e)

    # 图表
    plotly_js = _embed_plotlyjs()
    obj_chart = make_obj_bar_chart(obj_datasets)
    score_hist = make_score_histogram(subj.scores if subj else [], "主观评分分布")
    radar = make_radar_chart(subj.tag_dim_avg if subj else {})
    gen_chart = make_gen_metrics_chart(gen.case_metrics if gen else [])

    data = SingleRunData(
        run=run_info,
        obj_datasets=obj_datasets,
        subj=subj,
        badcases=badcases,
        gen=gen,
        ai_advice=ai_advice,
    )

    html = _render(
        "single_run.html.j2",
        data=data,
        plotly_js=plotly_js,
        obj_chart=obj_chart,
        score_hist=score_hist,
        radar=radar,
        gen_chart=gen_chart,
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(html, encoding="utf-8")
    logger.info("单次报告已生成: %s", output_path)


def build_compare_report(
    session: Session,
    model_name: str,
    last_n: int = 2,
    output_path: str = "report_compare.html",
    judge_config_path: str = "config/judge_config.yaml",
    with_ai_advice: bool = True,
) -> None:
    """生成多版本对比 HTML 报告，写入 output_path。"""
    # 获取最近 N 次（objective 或 subjective 均算），按时间从旧到新
    runs_all = session.scalars(
        select(EvalRun)
        .where(EvalRun.model_name == model_name)
        .order_by(EvalRun.created_at.desc())
        .limit(last_n)
    ).all()
    runs_all = list(reversed(runs_all))  # 从旧到新

    if not runs_all:
        raise ValueError(f"模型 '{model_name}' 无评测记录")

    compare_runs: list[CompareRunData] = []
    for run in runs_all:
        compare_runs.append(CompareRunData(
            run=_make_run_info(run),
            obj_datasets=_query_obj_datasets(session, run.id),
            subj=_query_subj_summary(session, run.id),
            badcases=_query_badcases(session, run.id),
            gen=_query_gen_summary(session, run.id),
        ))

    # delta：最新 vs 次新（如果有两个以上版本）
    obj_deltas: list[DeltaInfo] = []
    if len(compare_runs) >= 2:
        obj_deltas = _compute_deltas(
            compare_runs[-2].obj_datasets,
            compare_runs[-1].obj_datasets,
        )

    # AI 建议
    ai_advice = ""
    if with_ai_advice:
        try:
            from framework.evaluators.subjective.judge_clients import build_judge_client
            from framework.reporters.ai_advice import generate_compare_advice
            client = build_judge_client(judge_config_path)

            latest = compare_runs[-1]
            bc_examples = [
                {
                    "case_id": bc.case_id,
                    "question": bc.question,
                    "judge_reasoning": bc.judge_reasoning,
                    "tags": bc.tags,
                }
                for bc in latest.badcases
            ]
            ai_advice = generate_compare_advice(
                client,
                model_name=model_name,
                run_labels=[r.run.label for r in compare_runs],
                run_avg_scores=[r.subj.avg_score if r.subj else None for r in compare_runs],
                run_badcase_rates=[r.subj.badcase_rate if r.subj else 0.0 for r in compare_runs],
                delta_items=[
                    {"dataset": d.dataset, "delta": d.delta, "trend": d.trend}
                    for d in obj_deltas
                ],
                badcase_examples=bc_examples,
            )
        except Exception as e:
            logger.warning("AI 对比建议生成失败（不影响报告）: %s", e)

    # 图表
    plotly_js = _embed_plotlyjs()

    run_labels = [r.run.label for r in compare_runs]
    all_datasets = sorted({d.name for r in compare_runs for d in r.obj_datasets})
    scores_per_run = [{d.name: d.score for d in r.obj_datasets} for r in compare_runs]
    compare_bar = make_compare_bar_chart(run_labels, all_datasets, scores_per_run)

    avg_scores = [r.subj.avg_score if r.subj else None for r in compare_runs]
    badcase_rates = [r.subj.badcase_rate if r.subj else 0.0 for r in compare_runs]
    subj_trend = make_subj_trend_chart(run_labels, avg_scores, badcase_rates)

    # 最新版本的雷达图
    latest_subj = compare_runs[-1].subj
    radar = make_radar_chart(
        latest_subj.tag_dim_avg if latest_subj else {},
        title=f"最新版本能力雷达图 ({compare_runs[-1].run.label})",
    )

    gen_compare = make_gen_compare_chart(run_labels, [r.gen for r in compare_runs])

    data = CompareData(
        model_name=model_name,
        runs=compare_runs,
        obj_deltas=obj_deltas,
        ai_advice=ai_advice,
    )

    html = _render(
        "compare.html.j2",
        data=data,
        plotly_js=plotly_js,
        compare_bar=compare_bar,
        subj_trend=subj_trend,
        radar=radar,
        gen_compare=gen_compare,
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(html, encoding="utf-8")
    logger.info("对比报告已生成: %s", output_path)


# ---------------------------------------------------------------------------
# 会话聚合报告（一次 pytest 执行 → 一份报告）
# ---------------------------------------------------------------------------

def _combine_subj_summaries(summaries: list[SubjSummary]) -> Optional[SubjSummary]:
    """将多个数据集的 SubjSummary 合并为一个总览。"""
    summaries = [s for s in summaries if s is not None]
    if not summaries:
        return None

    total = sum(s.total_cases for s in summaries)
    badcase_count = sum(s.badcase_count for s in summaries)
    all_scores = [sc for s in summaries for sc in s.scores]

    dim_buckets: dict[str, list[float]] = defaultdict(list)
    tag_dim_buckets: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for s in summaries:
        for dim, avg in s.dimension_avg.items():
            dim_buckets[dim].extend([avg] * s.total_cases)
        for tag, dims in s.tag_dim_avg.items():
            for dim, avg in dims.items():
                tag_dim_buckets[tag][dim].append(avg)

    return SubjSummary(
        total_cases=total,
        badcase_count=badcase_count,
        badcase_rate=round(badcase_count / total, 3) if total else 0.0,
        avg_score=round(sum(all_scores) / len(all_scores), 2) if all_scores else None,
        scores=all_scores,
        dimension_avg={d: round(sum(v) / len(v), 2) for d, v in dim_buckets.items()},
        tag_dim_avg={
            t: {d: round(sum(v) / len(v), 2) for d, v in dims.items()}
            for t, dims in tag_dim_buckets.items()
        },
    )


def build_session_report(
    session: Session,
    runs: list[EvalRun],
    output_path: str,
) -> None:
    """将一次 pytest 会话产生的所有 EvalRun 聚合成一份 HTML 报告。"""
    runs_by_model: dict[str, list[EvalRun]] = defaultdict(list)
    for run in runs:
        runs_by_model[run.model_name].append(run)

    models_data: list[ModelSessionData] = []
    for model_name, model_runs in runs_by_model.items():
        obj_runs = [r for r in model_runs if r.run_type == "objective"]
        subj_runs = sorted([r for r in model_runs if r.run_type == "subjective"], key=lambda r: r.id)
        gen_runs = sorted([r for r in model_runs if r.run_type == "generation"], key=lambda r: r.id)
        safety_runs = sorted([r for r in model_runs if r.run_type == "safety"], key=lambda r: r.id)

        # 客观
        obj_datasets: list[ObjDataset] = []
        obj_run_id: Optional[int] = None
        if obj_runs:
            obj_run = max(obj_runs, key=lambda r: r.id)
            obj_run_id = obj_run.id
            obj_datasets = _query_obj_datasets(session, obj_run.id)

        # 主观（按数据集分组）
        subj_datasets: list[DatasetSubjData] = []
        subj_summaries: list[SubjSummary] = []
        all_badcases: list[BadcaseItem] = []
        for subj_run in subj_runs:
            summary = _query_subj_summary(session, subj_run.id)
            if summary:
                first = session.scalars(
                    select(SubjectiveResult)
                    .where(SubjectiveResult.run_id == subj_run.id)
                    .limit(1)
                ).first()
                ds_name = first.dataset_name if first else f"run_{subj_run.id}"
                subj_datasets.append(DatasetSubjData(
                    dataset_name=ds_name,
                    run_id=subj_run.id,
                    total_cases=summary.total_cases,
                    badcase_count=summary.badcase_count,
                    badcase_rate=summary.badcase_rate,
                    avg_score=summary.avg_score,
                ))
                subj_summaries.append(summary)
                all_badcases.extend(_query_badcases(session, subj_run.id))

        combined_subj = _combine_subj_summaries(subj_summaries)

        # 生成质量（按数据集分组）
        gen_datasets: list[DatasetGenData] = []
        for gen_run in gen_runs:
            gen_sum = _query_gen_summary(session, gen_run.id)
            if gen_sum:
                first = session.scalars(
                    select(GenerationResult)
                    .where(GenerationResult.run_id == gen_run.id)
                    .limit(1)
                ).first()
                ds_name = first.dataset_name if first else f"run_{gen_run.id}"
                gen_datasets.append(DatasetGenData(
                    dataset_name=ds_name,
                    run_id=gen_run.id,
                    total_cases=gen_sum.total_cases,
                    avg_bleu=gen_sum.avg_bleu,
                    avg_rouge1=gen_sum.avg_rouge1,
                    avg_rouge2=gen_sum.avg_rouge2,
                    avg_rougeL=gen_sum.avg_rougeL,
                    avg_recall=gen_sum.avg_recall,
                ))

        # 安全评测（按数据集分组）
        safety_datasets: list[DatasetSafetyData] = []
        all_unsafe_cases: list[SafetyUnsafeItem] = []
        for safety_run in safety_runs:
            rows = session.scalars(
                select(SafetyResult).where(SafetyResult.run_id == safety_run.id)
            ).all()
            if not rows:
                continue
            first = rows[0]
            ds_name = first.dataset_name
            total = len(rows)
            safe_count = sum(1 for r in rows if r.is_safe)
            unsafe_count = total - safe_count
            by_cat: dict[str, dict] = {}
            for r in rows:
                cat = r.category or "general"
                by_cat.setdefault(cat, {"total": 0, "safe": 0})
                by_cat[cat]["total"] += 1
                if r.is_safe:
                    by_cat[cat]["safe"] += 1
            safety_datasets.append(DatasetSafetyData(
                dataset_name=ds_name,
                run_id=safety_run.id,
                total_cases=total,
                safe_count=safe_count,
                unsafe_count=unsafe_count,
                safety_rate=round(safe_count / total, 3) if total else None,
                by_category={
                    cat: round(v["safe"] / v["total"], 3) if v["total"] else None
                    for cat, v in by_cat.items()
                },
            ))
            for r in rows:
                if not r.is_safe:
                    all_unsafe_cases.append(SafetyUnsafeItem(
                        case_id=r.case_id,
                        dataset_name=r.dataset_name,
                        category=r.category or "general",
                        question=r.question or "",
                        model_answer=r.model_answer or "",
                        judge_reasoning=r.judge_reasoning or "",
                        tags=r.tags or [],
                    ))

        models_data.append(ModelSessionData(
            model_name=model_name,
            obj_run_id=obj_run_id,
            obj_datasets=obj_datasets,
            subj_datasets=subj_datasets,
            combined_subj=combined_subj,
            badcases=all_badcases,
            gen_datasets=gen_datasets,
            safety_datasets=safety_datasets,
            unsafe_cases=all_unsafe_cases,
        ))

    session_ts = (
        min(r.created_at for r in runs).strftime("%Y-%m-%d %H:%M")
        if runs else datetime.utcnow().strftime("%Y-%m-%d %H:%M")
    )

    session_data = SessionData(
        session_ts=session_ts,
        total_runs=len(runs),
        models=models_data,
    )

    # 图表（每个模型一组）
    plotly_js = _embed_plotlyjs()
    model_charts: list[dict] = []
    for md in models_data:
        model_charts.append({
            "obj_chart": make_obj_bar_chart(md.obj_datasets),
            "radar": make_radar_chart(
                md.combined_subj.tag_dim_avg if md.combined_subj else {},
                title="能力维度雷达图",
            ),
            "hist": make_score_histogram(
                md.combined_subj.scores if md.combined_subj else [],
                title="主观评分分布",
            ),
        })

    html = _render(
        "session_report.html.j2",
        data=session_data,
        model_charts=model_charts,
        plotly_js=plotly_js,
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(html, encoding="utf-8")
    logger.info("会话报告已生成: %s", output_path)

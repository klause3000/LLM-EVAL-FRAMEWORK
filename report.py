#!/usr/bin/env python
"""
评测结果查询工具

用法：
  python report.py                          # 列出所有运行记录
  python report.py --model qwen2.5:7b-instruct          # 查看某模型最近两次对比
  python report.py --model qwen2.5:7b-instruct --last 3 # 最近三次对比
  python report.py --run 5                  # 查看某次运行的详细结果
  python report.py --run 5 --badcase        # 只看 badcase
"""

import argparse
import sys

from sqlalchemy import select

from framework.storage.db import init_db, get_session, get_badcase_summary
from framework.storage.models import EvalRun, ObjectiveResult, SubjectiveResult
from framework.reporters import print_full_report, build_single_run_report, build_compare_report


def cmd_list(session):
    """列出所有评测运行记录。"""
    runs = session.scalars(select(EvalRun).order_by(EvalRun.created_at.desc())).all()
    if not runs:
        print("暂无评测记录。")
        return

    print(f"\n{'Run ID':<8}{'模型':<28}{'版本':<18}{'类型':<12}{'时间'}")
    print("-" * 80)
    for r in runs:
        print(
            f"{r.id:<8}{r.model_name:<28}{(r.model_version or '-'):<18}"
            f"{r.run_type:<12}{r.created_at.strftime('%Y-%m-%d %H:%M:%S')}"
        )
    print()


def cmd_run_detail(session, run_id: int, only_badcase: bool = False, full: bool = False):
    """显示某次运行的详细结果。"""
    run = session.get(EvalRun, run_id)
    if not run:
        print(f"Run #{run_id} 不存在。")
        return

    print(f"\n{'=' * 80}")
    print(f"  Run #{run.id}  |  {run.model_name}  |  v={run.model_version or '-'}")
    print(f"  类型: {run.run_type}  |  时间: {run.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # 客观评测结果
    obj_results = session.scalars(
        select(ObjectiveResult).where(ObjectiveResult.run_id == run_id)
    ).all()
    if obj_results:
        print("\n【客观评测】")
        print(f"  {'数据集':<20}{'得分':<12}{'样本数'}")
        print("  " + "-" * 40)
        for r in obj_results:
            score_str = f"{r.score * 100:.1f}%" if r.score is not None else "失败"
            err = f"  ERROR: {r.error[:60]}" if r.error else ""
            print(f"  {r.dataset_name:<20}{score_str:<12}{r.num_samples}{err}")

    # 主观评测结果
    subj_q = select(SubjectiveResult).where(SubjectiveResult.run_id == run_id)
    if only_badcase:
        subj_q = subj_q.where(SubjectiveResult.is_badcase == True)  # noqa: E712
    subj_q = subj_q.order_by(SubjectiveResult.dataset_name, SubjectiveResult.case_id)
    subj_results = session.scalars(subj_q).all()

    if subj_results:
        summary = get_badcase_summary(session, run_id)
        label = "【主观评测 — 仅 Badcase】" if only_badcase else "【主观评测】"
        print(f"\n{label}")
        print(
            f"  总 case: {summary['total_cases']}  |  "
            f"badcase: {summary['badcase_count']} ({summary['badcase_rate']*100:.0f}%)  |  "
            f"均分: {summary['avg_overall_score']}"
        )
        print()

        current_ds = None
        for r in subj_results:
            if r.dataset_name != current_ds:
                current_ds = r.dataset_name
                print(f"  数据集: {current_ds}")
                print(f"  {'Case ID':<16}{'均分':<8}{'Badcase':<10}{'维度得分'}")
                print("  " + "-" * 72)

            dim_str = "  ".join(f"{k}:{v}" for k, v in (r.dimension_scores or {}).items())
            flag = "✗ BAD" if r.is_badcase else "✓ OK "
            print(f"  {r.case_id:<16}{(r.overall_score or 0):<8.2f}{flag:<10}{dim_str}")
            print(f"    问题: {r.question[:80]}")
            if full and r.model_answer:
                print(f"    回答: {r.model_answer}")
            if r.judge_reasoning:
                print(f"    理由: {r.judge_reasoning[:100]}")
            print()
    elif only_badcase:
        print("\n  该次运行无 badcase 记录。\n")

    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="LLM 评测结果查询工具")
    parser.add_argument("--model", "-m", help="查看指定模型的版本对比")
    parser.add_argument("--last", "-n", type=int, default=2, help="对比最近 N 次（默认 2）")
    parser.add_argument("--run", "-r", type=int, help="查看指定 run_id 的详细结果")
    parser.add_argument("--badcase", "-b", action="store_true", help="只显示 badcase")
    parser.add_argument("--full", "-f", action="store_true", help="显示完整模型回答和评判理由（不截断）")
    parser.add_argument("--html", metavar="OUTPUT_PATH", help="生成 HTML 报告并保存到指定路径（需搭配 --run 或 --model）")
    parser.add_argument("--no-ai-advice", action="store_true", default=False, help="跳过 AI 优化建议生成（速度更快）")
    args = parser.parse_args()

    init_db()
    session = get_session()

    try:
        if args.html:
            if args.run:
                build_single_run_report(
                    session,
                    run_id=args.run,
                    output_path=args.html,
                    with_ai_advice=not args.no_ai_advice,
                )
                print(f"HTML 报告已生成: {args.html}")
            elif args.model:
                build_compare_report(
                    session,
                    model_name=args.model,
                    last_n=args.last,
                    output_path=args.html,
                    with_ai_advice=not args.no_ai_advice,
                )
                print(f"HTML 对比报告已生成: {args.html}")
            else:
                print("错误: --html 需要搭配 --run <id> 或 --model <name> 使用", file=sys.stderr)
                sys.exit(1)
        elif args.run:
            cmd_run_detail(session, args.run, only_badcase=args.badcase, full=args.full)
        elif args.model:
            print_full_report(session, model_name=args.model, last_n=args.last)
        else:
            cmd_list(session)
    finally:
        session.close()


if __name__ == "__main__":
    main()

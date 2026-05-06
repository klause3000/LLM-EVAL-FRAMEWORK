"""
主观评测测试：LLM-as-judge 对被测模型打分，结果入库。

运行方式：
  pytest tests/test_subjective.py -m subjective        # 全部主观评测
  pytest tests/test_subjective.py -k qwen2.5:7b        # 指定模型
  pytest tests/test_subjective.py -k qa_general        # 指定数据集
"""

import yaml
import pytest

from framework.evaluators.subjective import run_subjective_eval, load_subjective_datasets
from framework.storage.db import save_subjective_results
from framework.reporters import print_subjective_comparison

# 在模块级读取，供 parametrize 使用
with open("config/models.yaml", "r", encoding="utf-8") as _f:
    _MODELS: list[dict] = yaml.safe_load(_f)["models"]

_DATASETS: list[dict] = load_subjective_datasets("config/subjective_datasets")


@pytest.mark.subjective
@pytest.mark.parametrize("model_cfg", _MODELS, ids=[m["name"] for m in _MODELS])
@pytest.mark.parametrize(
    "dataset",
    _DATASETS,
    ids=[d.get("dataset_name", f"ds_{i}") for i, d in enumerate(_DATASETS)],
)
def test_subjective_eval(model_cfg: dict, dataset: dict, db_session):
    """
    对每个「模型 × 数据集」组合进行主观评测：
    1. 调用被测模型获取回答
    2. judge 模型对回答评分
    3. 结果入库
    4. 打印 badcase 覆盖报告
    """
    model_name: str = model_cfg["name"]
    model_version: str = model_cfg.get("version", "")
    dataset_name: str = dataset.get("dataset_name", "unknown")
    cases: list[dict] = dataset.get("cases", [])

    assert cases, f"数据集 {dataset_name} 中没有 case，请检查 YAML 文件"

    case_results = run_subjective_eval(
        model_name=model_name,
        cases=cases,
        dataset_name=dataset_name,
    )

    assert case_results, f"模型 {model_name} 在数据集 {dataset_name} 上未返回任何结果"

    # 写入数据库
    run = save_subjective_results(
        session=db_session,
        model_name=model_name,
        model_version=model_version,
        case_results=case_results,
    )

    # 打印版本对比报告
    print_subjective_comparison(db_session, model_name, last_n=2)

    # 统计 badcase
    badcases = [r for r in case_results if r.is_badcase]
    total = len(case_results)
    bad_rate = len(badcases) / total if total else 0

    # 打印 badcase 明细（便于 CI 日志查看）
    if badcases:
        print(f"\n[{model_name}][{dataset_name}] Badcase 明细（{len(badcases)}/{total}）:")
        for r in badcases:
            print(f"  - {r.case_id}: 均分={r.overall_score}  维度={r.dimension_scores}")
            print(f"    问题: {r.question[:80]}")
            print(f"    理由: {r.judge_reasoning[:100]}")

    # 断言：judge 打分流程本身没有全部崩溃（overall_score=0 代表异常）
    scored = [r for r in case_results if r.overall_score > 0]
    assert scored, (
        f"模型 {model_name} 在 {dataset_name} 上所有 case 的 judge 评分均为 0，"
        "请检查 judge 配置（API key / provider）是否正确。"
    )

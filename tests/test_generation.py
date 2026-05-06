"""
生成质量评测测试：对数据集中有 reference_answer 的 case 计算 BLEU/ROUGE/召回率。

运行方式：
  pytest tests/test_generation.py -m generation -v
  pytest tests/test_generation.py -k "qwen2.5:7b" -v
  pytest tests/test_generation.py -k "python_exception" -v
"""

import yaml
import pytest

from framework.evaluators.generation import run_generation_eval
from framework.evaluators.subjective.dataset_loader import load_subjective_datasets
from framework.storage.db import save_generation_results, get_generation_summary

with open("config/models.yaml", "r", encoding="utf-8") as _f:
    _MODELS: list[dict] = yaml.safe_load(_f)["models"]

_DATASETS = load_subjective_datasets()
# 只保留有 reference_answer 的数据集
_DATASETS = [
    ds for ds in _DATASETS
    if any(c.get("reference_answer") for c in ds.get("cases", []))
]

_PARAMS = [
    (m, ds)
    for m in _MODELS
    for ds in _DATASETS
]
_IDS = [f"{ds['dataset_name']}-{m['name']}" for m, ds in _PARAMS]


@pytest.mark.generation
@pytest.mark.parametrize("model_cfg,dataset", _PARAMS, ids=_IDS)
def test_generation_eval(model_cfg: dict, dataset: dict, db_session):
    model_name: str = model_cfg["name"]
    model_version: str = model_cfg.get("version", "")
    dataset_name: str = dataset["dataset_name"]
    cases: list[dict] = dataset["cases"]

    results = run_generation_eval(
        model_name=model_name,
        cases=cases,
        dataset_name=dataset_name,
    )

    assert results, f"模型 {model_name} 在数据集 {dataset_name} 上未返回任何结果"

    run = save_generation_results(
        session=db_session,
        model_name=model_name,
        model_version=model_version,
        case_results=results,
    )

    summary = get_generation_summary(db_session, run.id)

    print(f"\n{'='*60}")
    print(f"  生成质量评测 | {model_name} | {dataset_name}")
    print(f"{'='*60}")
    print(f"  总 case 数 : {summary['total_cases']}")
    print(f"  BLEU       : {summary.get('avg_bleu')}")
    print(f"  ROUGE-1    : {summary.get('avg_rouge1')}")
    print(f"  ROUGE-2    : {summary.get('avg_rouge2')}")
    print(f"  ROUGE-L    : {summary.get('avg_rougeL')}")
    print(f"  召回率     : {summary.get('avg_recall')}")
    print(f"{'='*60}\n")

    # 基本断言：至少有 ROUGE-L 分数返回
    assert summary.get("avg_rougeL") is not None

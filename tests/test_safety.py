"""
安全性评测测试：使用 CValues 格式数据集，通过 LLM-as-judge 判定 SAFE/UNSAFE。

运行方式：
  pytest tests/test_safety.py -m safety -v
  pytest tests/test_safety.py -k "qwen2.5:7b-instruct" -v
  pytest tests/test_safety.py -k "cvalues_sample" -v
"""

import yaml
import pytest

from framework.evaluators.safety import run_safety_eval
from framework.storage.db import save_safety_results, get_safety_summary

with open("config/models.yaml", "r", encoding="utf-8") as _f:
    _MODELS: list[dict] = yaml.safe_load(_f)["models"]


def _load_safety_datasets(path: str = "config/safety_datasets") -> list[dict]:
    from pathlib import Path
    datasets = []
    for f in sorted(Path(path).glob("*.yaml")):
        with open(f, "r", encoding="utf-8") as fp:
            datasets.append(yaml.safe_load(fp))
    return datasets


_DATASETS = _load_safety_datasets()
_PARAMS = [(m, ds) for m in _MODELS for ds in _DATASETS]
_IDS = [f"{ds['dataset_name']}-{m['name']}" for m, ds in _PARAMS]


@pytest.mark.safety
@pytest.mark.parametrize("model_cfg,dataset", _PARAMS, ids=_IDS)
def test_safety_eval(model_cfg: dict, dataset: dict, db_session):
    model_name: str = model_cfg["name"]
    model_version: str = model_cfg.get("version", "")
    dataset_name: str = dataset["dataset_name"]
    cases: list[dict] = dataset["cases"]

    results = run_safety_eval(
        model_name=model_name,
        cases=cases,
        dataset_name=dataset_name,
    )

    assert results, f"模型 {model_name} 在数据集 {dataset_name} 上未返回任何安全评测结果"

    run = save_safety_results(
        session=db_session,
        model_name=model_name,
        model_version=model_version,
        case_results=results,
    )

    summary = get_safety_summary(db_session, run.id)

    print(f"\n{'='*60}")
    print(f"  安全性评测 | {model_name} | {dataset_name}")
    print(f"{'='*60}")
    print(f"  总 case 数 : {summary['total_cases']}")
    print(f"  安全率     : {summary.get('safety_rate', 0) * 100:.1f}%")
    print(f"  安全 case  : {summary.get('safe_count', 0)}")
    print(f"  不安全 case: {summary.get('unsafe_count', 0)}")
    print(f"  分类安全率 :")
    for cat, rate in (summary.get("by_category") or {}).items():
        rate_str = f"{rate * 100:.1f}%" if rate is not None else "N/A"
        print(f"    {cat:<20} {rate_str}")
    print(f"{'='*60}\n")

    # 基本断言：有结果返回即通过（安全率由报告决策，不强制阈值）
    assert summary.get("total_cases", 0) > 0

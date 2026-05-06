"""
客观评测测试：使用 evalscope 对标准数据集评测，结果入库。

运行方式：
  pytest tests/test_objective.py -m objective          # 全部客观评测
  pytest tests/test_objective.py -m "objective and slow"  # 完整数据集（慢）
  pytest tests/test_objective.py -k qwen2.5:7b         # 指定模型
"""

import warnings

import yaml
import pytest

from framework.evaluators.objective import run_objective_eval
from framework.storage.db import save_objective_results
from framework.reporters import print_objective_comparison

# 包含这些关键词的错误视为网络/缓存问题，不触发 hard fail
_NETWORK_ERR_KEYWORDS = (
    "cache", "response ended", "couldn't be found",
    "no (supported) data files", "connection", "timeout",
)

# 在模块级读取 models.yaml，供 parametrize 使用
with open("config/models.yaml", "r", encoding="utf-8") as _f:
    _MODELS: list[dict] = yaml.safe_load(_f)["models"]


@pytest.mark.objective
@pytest.mark.slow
@pytest.mark.parametrize("model_cfg", _MODELS, ids=[m["name"] for m in _MODELS])
def test_objective_eval(model_cfg: dict, db_session):
    """
    对每个模型运行全量客观评测，结果入库，并断言至少有得分返回。
    """
    model_name: str = model_cfg["name"]
    model_version: str = model_cfg.get("version", "")

    results = run_objective_eval(
        model_name=model_name,
        model_version=model_version,
    )

    # 写入数据库
    run = save_objective_results(
        session=db_session,
        model_name=model_name,
        model_version=model_version,
        results=results,
    )

    # 打印版本对比报告
    print_objective_comparison(db_session, model_name, last_n=2)

    # 基本断言：至少有一个数据集返回了结果
    assert results, f"模型 {model_name} 未返回任何客观评测结果"

    # 按错误类型分类：网络/缓存类只 warn，其余 hard fail
    failed = {ds: r.get("error") for ds, r in results.items() if r.get("error") and r.get("score") is None}
    hard_failed, soft_failed = {}, {}
    for ds, err in failed.items():
        err_lower = (err or "").lower()
        if any(kw in err_lower for kw in _NETWORK_ERR_KEYWORDS):
            soft_failed[ds] = err
        else:
            hard_failed[ds] = err

    if soft_failed:
        for ds, err in soft_failed.items():
            warnings.warn(f"数据集 {ds} 因网络/缓存问题跳过: {err[:120]}", UserWarning, stacklevel=2)

    if hard_failed:
        detail = "\n".join(f"  [{ds}] {err}" for ds, err in hard_failed.items())
        pytest.fail(
            f"以下数据集评测失败:\n{detail}\n"
            "请检查 evalscope 版本兼容性和 ollama 端点是否可达。"
        )

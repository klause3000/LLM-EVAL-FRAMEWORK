"""
pytest 全局 fixture：
- ollama 服务就绪检查 + 模型自动拉取（session 级，仅运行一次）
- 数据库 session
- models_config / subjective_datasets 共用数据
- 会话结束后自动生成聚合 HTML 报告
"""

from datetime import datetime
from pathlib import Path

import yaml
import pytest

from framework.model_manager.ollama_manager import pull_all_models, wait_for_ollama
from framework.evaluators.subjective.dataset_loader import load_subjective_datasets
from framework.storage.db import init_db, get_session

# conftest.py 被 import 时即记录会话开始时间（早于所有测试执行）
_SESSION_START_UTC: datetime = datetime.utcnow()


# ---------------------------------------------------------------------------
# Ollama + 模型拉取（整个测试会话只运行一次）
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def ensure_ollama_and_models():
    """确保 ollama 服务可用，并按 models.yaml 拉取所有模型。"""
    assert wait_for_ollama(retries=5, interval=3), (
        "ollama 服务未就绪，请先启动 ollama（ollama serve）"
    )
    results = pull_all_models("config/models.yaml")
    failed = [name for name, ok in results.items() if not ok]
    assert not failed, f"以下模型拉取失败，请检查网络或模型名称: {failed}"


# ---------------------------------------------------------------------------
# 数据库
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def db_engine():
    """初始化数据库（SQLite），返回 engine。"""
    return init_db()


@pytest.fixture
def db_session(db_engine):
    """每个测试函数独立 session，测试结束后自动关闭。"""
    session = get_session()
    yield session
    session.close()


# ---------------------------------------------------------------------------
# 配置数据
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def models_config() -> list[dict]:
    """返回 models.yaml 中的模型列表。"""
    with open("config/models.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["models"]


@pytest.fixture(scope="session")
def subjective_datasets() -> list[dict]:
    """返回所有主观评测数据集。"""
    return load_subjective_datasets("config/subjective_datasets")


# ---------------------------------------------------------------------------
# 会话结束钩子：自动生成聚合 HTML 报告
# ---------------------------------------------------------------------------

def pytest_sessionfinish(session, exitstatus):
    """所有测试跑完后，将本次会话产生的所有 EvalRun 聚合成一份 HTML 报告。"""
    try:
        from sqlalchemy import select
        from framework.storage.db import init_db, get_session as _get_session
        from framework.storage.models import EvalRun
        from framework.reporters.html_builder import build_session_report

        init_db()
        db = _get_session()
        try:
            runs = list(db.scalars(
                select(EvalRun)
                .where(EvalRun.created_at >= _SESSION_START_UTC)
                .order_by(EvalRun.id)
            ))
            if not runs:
                return

            ts = _SESSION_START_UTC.strftime("%Y%m%d_%H%M%S")
            output = str(Path("reports") / f"session_{ts}.html")
            build_session_report(db, runs, output)

            # 打印到终端
            tw = session.config.pluginmanager.get_plugin("terminalreporter")
            if tw:
                tw.write_sep("-", f"HTML 报告已生成: {Path(output).resolve()}", bold=True)
        finally:
            db.close()
    except Exception as exc:
        print(f"\n[会话报告] 生成失败: {exc}")

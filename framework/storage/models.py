"""
SQLAlchemy ORM 模型：存储客观/主观评测结果，支持版本间对比。
"""

from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class EvalRun(Base):
    """一次评测运行（一个模型 × 一次触发）。"""

    __tablename__ = "eval_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(100), nullable=False, index=True)
    model_version = Column(String(50))
    run_type = Column(String(20), nullable=False)  # "objective" | "subjective" | "generation" | "all"
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    objective_results = relationship(
        "ObjectiveResult", back_populates="run", cascade="all, delete-orphan"
    )
    subjective_results = relationship(
        "SubjectiveResult", back_populates="run", cascade="all, delete-orphan"
    )
    generation_results = relationship(
        "GenerationResult", back_populates="run", cascade="all, delete-orphan"
    )
    safety_results = relationship(
        "SafetyResult", back_populates="run", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return (
            f"<EvalRun id={self.id} model={self.model_name} "
            f"type={self.run_type} at={self.created_at}>"
        )


class ObjectiveResult(Base):
    """客观评测结果（per 模型 × dataset）。"""

    __tablename__ = "objective_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey("eval_runs.id", ondelete="CASCADE"), nullable=False, index=True)
    dataset_name = Column(String(100), nullable=False, index=True)
    score = Column(Float)           # 准确率 0~1
    num_samples = Column(Integer, default=0)
    raw_metrics = Column(JSON)      # evalscope 原始输出
    error = Column(Text)            # 若评测失败，记录异常信息
    created_at = Column(DateTime, default=datetime.utcnow)

    run = relationship("EvalRun", back_populates="objective_results")

    def __repr__(self) -> str:
        return (
            f"<ObjectiveResult dataset={self.dataset_name} "
            f"score={self.score} run_id={self.run_id}>"
        )


class SubjectiveResult(Base):
    """主观评测结果（per case）。"""

    __tablename__ = "subjective_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey("eval_runs.id", ondelete="CASCADE"), nullable=False, index=True)
    dataset_name = Column(String(100), nullable=False, index=True)
    case_id = Column(String(100), nullable=False, index=True)
    question = Column(Text, nullable=False)
    model_answer = Column(Text)
    reference_answer = Column(Text)
    dimension_scores = Column(JSON)     # {"生成质量": 8, "逻辑": 7, ...}
    overall_score = Column(Float)
    judge_reasoning = Column(Text)
    is_badcase = Column(Boolean, default=False, index=True)
    tags = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

    run = relationship("EvalRun", back_populates="subjective_results")

    def __repr__(self) -> str:
        return (
            f"<SubjectiveResult case={self.case_id} "
            f"score={self.overall_score} badcase={self.is_badcase}>"
        )


class GenerationResult(Base):
    """生成质量评测结果（per case）——BLEU / ROUGE / 召回率。"""

    __tablename__ = "generation_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey("eval_runs.id", ondelete="CASCADE"), nullable=False, index=True)
    dataset_name = Column(String(100), nullable=False, index=True)
    case_id = Column(String(100), nullable=False, index=True)
    question = Column(Text, nullable=False)
    model_answer = Column(Text)
    reference_answer = Column(Text)
    bleu = Column(Float)       # 0~100
    rouge1 = Column(Float)     # 0~1，F1
    rouge2 = Column(Float)     # 0~1，F1
    rougeL = Column(Float)     # 0~1，F1
    recall = Column(Float)     # 0~1，token-level recall
    tags = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

    run = relationship("EvalRun", back_populates="generation_results")

    def __repr__(self) -> str:
        return (
            f"<GenerationResult case={self.case_id} "
            f"bleu={self.bleu} rouge1={self.rouge1}>"
        )


class SafetyResult(Base):
    """安全性评测结果（per case）——CValues 格式。"""

    __tablename__ = "safety_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey("eval_runs.id", ondelete="CASCADE"), nullable=False, index=True)
    dataset_name = Column(String(100), nullable=False, index=True)
    case_id = Column(String(100), nullable=False, index=True)
    category = Column(String(100))        # CValues 分类：physical_harm / illegal / ...
    question = Column(Text, nullable=False)
    model_answer = Column(Text)
    is_safe = Column(Boolean, index=True) # True=SAFE, False=UNSAFE
    judge_reasoning = Column(Text)
    tags = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

    run = relationship("EvalRun", back_populates="safety_results")

    def __repr__(self) -> str:
        return (
            f"<SafetyResult case={self.case_id} "
            f"category={self.category} is_safe={self.is_safe}>"
        )

# app/models/experiments.py
import uuid
from datetime import datetime
from sqlalchemy import (
    Column, String, Text, DateTime, ForeignKey, Boolean, Integer, Numeric, JSON
)
from sqlalchemy.orm import relationship
from sqlalchemy import CheckConstraint, UniqueConstraint
from app.db.base import Base
from app.models._types import GUID  # portable UUID type (Postgres+SQLite)

# ===== Core entities =====
class Experiment(Base):
    __tablename__ = "experiments"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    name = Column(Text, nullable=False)
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    runs = relationship("ExperimentRun", back_populates="experiment", cascade="all, delete-orphan")


class ExperimentRun(Base):
    __tablename__ = "experiment_runs"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    experiment_id = Column(GUID(), ForeignKey("experiments.id", ondelete="CASCADE"), nullable=False)
    model_name = Column(String)
    params = Column(JSON)     # hyperparams, temp, etc.
    notes = Column(Text)
    started_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    ended_at = Column(DateTime)
    experiment = relationship("Experiment", back_populates="runs")
    items = relationship("RunItem", back_populates="run", cascade="all, delete-orphan")
    usefulness_ratings = relationship("UsefulnessRating", back_populates="run", cascade="all, delete-orphan")
    decision_support_metrics = relationship("DecisionSupportMetric", back_populates="run", cascade="all, delete-orphan")


# ===== Per-run-item outputs =====
class RunItem(Base):
    __tablename__ = "run_items"
    __table_args__ = (
        UniqueConstraint("run_id", "dataset_id", name="uq_run_items_run_dataset"),
    )

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    run_id = Column(GUID(), ForeignKey("experiment_runs.id", ondelete="CASCADE"), nullable=False)
    dataset_id = Column(Integer, ForeignKey("evaluation_dataset.id", ondelete="CASCADE"), nullable=False)
    predicted_plan = Column(JSON)   # normalized list of task names (no args), e.g., ["predict_population_linear","plot_population"]
    response_text = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    run = relationship("ExperimentRun", back_populates="items")
    dataset = relationship("EvaluationDataset")
    planning_eval = relationship("PlanningGpt4Eval", back_populates="run_item", uselist=False, cascade="all, delete-orphan")
    task_eval = relationship("TaskEval", back_populates="run_item", uselist=False, cascade="all, delete-orphan")
    feedback = relationship("AgentRunFeedback", back_populates="run_item", uselist=False, cascade="all, delete-orphan")
    interpretability_ratings = relationship("InterpretabilityRating", back_populates="run_item", cascade="all, delete-orphan")


# ===== Planning-level metric (GPT-4 judge) =====
class PlanningGpt4Eval(Base):
    __tablename__ = "planning_gpt4_eval"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    run_item_id = Column(GUID(), ForeignKey("run_items.id", ondelete="CASCADE"), nullable=False, unique=True)
    judge_model = Column(String, default="gpt-4")
    prompt_used = Column(Text)
    raw_response = Column(JSON)     # raw JSON from judge
    choice = Column(Boolean)        # True = "yes", False = "no"
    reason = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    run_item = relationship("RunItem", back_populates="planning_eval")


# ===== Task-level evaluation =====
class TaskEval(Base):
    __tablename__ = "task_eval"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    run_item_id = Column(GUID(), ForeignKey("run_items.id", ondelete="CASCADE"), nullable=False, unique=True)
    gold_task_sequence = Column(JSON, nullable=False)       # e.g., ["predict_population_linear","plot_population"]
    predicted_task_sequence = Column(JSON, nullable=False)  # same format
    exact_match = Column(Boolean, nullable=False, default=False)
    tp = Column(Integer, nullable=False, default=0)
    fp = Column(Integer, nullable=False, default=0)
    fn = Column(Integer, nullable=False, default=0)
    f1 = Column(Numeric(6, 4))  # per-item F1
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    run_item = relationship("RunItem", back_populates="task_eval")


# ===== System-level metrics =====
class AgentRunFeedback(Base):
    __tablename__ = "agent_run_feedback"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    run_item_id = Column(GUID(), ForeignKey("run_items.id", ondelete="CASCADE"), nullable=False, unique=True)
    approved = Column(Boolean)  # liked/disliked
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    run_item = relationship("RunItem", back_populates="feedback")


class InterpretabilityRating(Base):
    __tablename__ = "interpretability_ratings"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    run_item_id = Column(GUID(), ForeignKey("run_items.id", ondelete="CASCADE"), nullable=False)
    rater_id = Column(String)
    clarity = Column(Integer)        # 1..5
    completeness = Column(Integer)   # 1..5
    correctness = Column(Integer)    # 1..5
    comments = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    __table_args__ = (
        CheckConstraint("clarity BETWEEN 1 AND 5", name="ck_interp_clarity"),
        CheckConstraint("completeness BETWEEN 1 AND 5", name="ck_interp_completeness"),
        CheckConstraint("correctness BETWEEN 1 AND 5", name="ck_interp_correctness"),
    )
    run_item = relationship("RunItem", back_populates="interpretability_ratings")


class UsefulnessRating(Base):
    __tablename__ = "usefulness_ratings"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    run_id = Column(GUID(), ForeignKey("experiment_runs.id", ondelete="CASCADE"), nullable=False)
    rater_id = Column(String)
    usefulness = Column(Integer)  # 1..5
    comments = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    __table_args__ = (
        CheckConstraint("usefulness BETWEEN 1 AND 5", name="ck_usefulness_score"),
    )
    run = relationship("ExperimentRun", back_populates="usefulness_ratings")


class DecisionSupportMetric(Base):
    __tablename__ = "decision_support_metrics"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    run_id = Column(GUID(), ForeignKey("experiment_runs.id", ondelete="CASCADE"), nullable=False)
    baseline_tool = Column(String)    # e.g., "BI-tool", "human-only"
    effectiveness_pct = Column(Numeric(6, 2))  # improvement %
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    run = relationship("ExperimentRun", back_populates="decision_support_metrics")


class SusSurvey(Base):
    __tablename__ = "sus_surveys"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    experiment_id = Column(GUID(), ForeignKey("experiments.id", ondelete="CASCADE"), nullable=False)
    respondent_id = Column(String)
    submitted_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    s1 = Column(Integer); s2 = Column(Integer); s3 = Column(Integer); s4 = Column(Integer); s5 = Column(Integer)
    s6 = Column(Integer); s7 = Column(Integer); s8 = Column(Integer); s9 = Column(Integer); s10 = Column(Integer)
    sus_score = Column(Numeric(6, 2))  # precomputed score (0–100)
    __table_args__ = (
        CheckConstraint("s1 BETWEEN 1 AND 5", name="ck_sus_s1"),
        CheckConstraint("s2 BETWEEN 1 AND 5", name="ck_sus_s2"),
        CheckConstraint("s3 BETWEEN 1 AND 5", name="ck_sus_s3"),
        CheckConstraint("s4 BETWEEN 1 AND 5", name="ck_sus_s4"),
        CheckConstraint("s5 BETWEEN 1 AND 5", name="ck_sus_s5"),
        CheckConstraint("s6 BETWEEN 1 AND 5", name="ck_sus_s6"),
        CheckConstraint("s7 BETWEEN 1 AND 5", name="ck_sus_s7"),
        CheckConstraint("s8 BETWEEN 1 AND 5", name="ck_sus_s8"),
        CheckConstraint("s9 BETWEEN 1 AND 5", name="ck_sus_s9"),
        CheckConstraint("s10 BETWEEN 1 AND 5", name="ck_sus_s10"),
    )

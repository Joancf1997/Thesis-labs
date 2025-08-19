import uuid
from datetime import datetime
from sqlalchemy import (
    Column, String, Text, DateTime, ForeignKey, Boolean, Integer, Numeric, JSON
)
from sqlalchemy.orm import relationship
from app.db.base import Base
from app.models._types import GUID


class Experiment(Base):
    __tablename__ = "experiments"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    name = Column(Text, nullable=False)
    description = Column(Text)
    active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    runs_map = relationship("ExperimentRun", back_populates="experiment", cascade="all, delete-orphan")


class ExperimentRun(Base):
    """
    Optional mapping row that pairs an agent_run with an experiment.
    If you add experiment_id directly on AgentRun, this becomes optional.
    """
    __tablename__ = "experiment_runs"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    experiment_id = Column(GUID(), ForeignKey("experiments.id", ondelete="CASCADE"), nullable=False)
    run_id = Column(GUID(), unique=True, nullable=False)  # FK to agent_runs.id (enforced app-side)
    dataset_id = Column(Integer, ForeignKey("evaluation_dataset.id", ondelete="CASCADE"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    experiment = relationship("Experiment", back_populates="runs_map")
    dataset = relationship("EvaluationDataset")


class PlanningGpt4Eval(Base):
    __tablename__ = "planning_gpt4_eval"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    run_id = Column(GUID(), unique=True, nullable=False)  # FK to agent_runs.id (enforced app-side)

    judge_model = Column(String, default="gpt-4")
    prompt_used = Column(Text)
    raw_response = Column(JSON)
    choice = Column(Boolean)    # True = "yes"
    reason = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class TaskEval(Base):
    __tablename__ = "task_eval"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    run_id = Column(GUID(), unique=True, nullable=False)  # FK to agent_runs.id (enforced app-side)

    gold_task_sequence = Column(JSON)            # null in production
    predicted_task_sequence = Column(JSON, nullable=False)

    tp = Column(Integer, default=0, nullable=False)
    fp = Column(Integer, default=0, nullable=False)
    fn = Column(Integer, default=0, nullable=False)
    f1 = Column(Numeric(6, 4))
    precision = Column(Numeric(6, 4))
    recall = Column(Numeric(6, 4))
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class UserFeedback(Base):
    """
    Unified user feedback per run: covers approval and usefulness (Likert).
    """
    __tablename__ = "user_feedback"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    run_id = Column(GUID(), unique=True, nullable=False)  # FK to agent_runs.id (enforced app-side)

    approved = Column(Boolean)
    usefulness = Column(Integer)  # 1..5
    comments = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class InterpretabilityRating(Base):
    __tablename__ = "interpretability_ratings"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    run_id = Column(GUID(), nullable=False)  # FK to agent_runs.id (enforced app-side)
    rater_id = Column(String)

    clarity = Column(Integer)        # 1..5
    completeness = Column(Integer)   # 1..5
    correctness = Column(Integer)    # 1..5
    comments = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class DecisionSupportMetric(Base):
    __tablename__ = "decision_support_metrics"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    run_id = Column(GUID(), unique=True, nullable=False)  # FK to agent_runs.id (enforced app-side)

    baseline_tool = Column(String)
    effectiveness_pct = Column(Numeric(6, 2))
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class SusSurvey(Base):
    __tablename__ = "sus_surveys"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    experiment_id = Column(GUID(), ForeignKey("experiments.id", ondelete="CASCADE"), nullable=False)
    respondent_id = Column(String)
    submitted_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    s1 = Column(Integer); s2 = Column(Integer); s3 = Column(Integer); s4 = Column(Integer); s5 = Column(Integer)
    s6 = Column(Integer); s7 = Column(Integer); s8 = Column(Integer); s9 = Column(Integer); s10 = Column(Integer)
    sus_score = Column(Numeric(6, 2))

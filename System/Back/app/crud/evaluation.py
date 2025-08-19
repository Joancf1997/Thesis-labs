from datetime import datetime
from sqlalchemy.orm import Session
from ..models.agentRun import AgentRun
from ..models.agentStep import AgentStep
from ..models.llmCall import LLMCall
from ..models.log import Log
from ..models.agentRun import AgentRun
from ..db.session import SessionLocal
from ..models.message import Message
from ..models.userSession import UserSession
from ..models.toolCall import ToolCall
from sqlalchemy.dialects.postgresql import UUID

from app.models.experiments import (
    Experiment, ExperimentRun,
    PlanningGpt4Eval, TaskEval, UserFeedback, InterpretabilityRating
)


def start_agent_evaluation(db: Session, experiment: Experiment) -> Experiment:
    db.add(experiment)    
    db.commit()
    db.refresh(experiment)
    return experiment


def create_experiment_run(db: Session, exp_run: ExperimentRun):
    db.add(exp_run)    
    db.commit()


def create_task_evaluation(db: Session, task_val: TaskEval): 
    db.add(task_val)    
    db.commit()

def create_gptJudge_evaluation(db: Session, judeEval: PlanningGpt4Eval): 
    db.add(judeEval)    
    db.commit()
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




#  == Session == 
def start_user_session(db: Session, user_id: UUID) -> UUID:
    s = UserSession(user_id=user_id)
    db.add(s)
    db.commit()
    db.refresh(s)
    return s.id

#  == Agent Run == 
def create_agent_run(db: Session, run: AgentRun) -> AgentRun:
    db.add(run)    
    db.commit()
    db.refresh(run)
    return run

def end_agent_run(db: Session, agent_run: AgentRun): 
    agent_run.status = "completed"
    agent_run.ended_at = datetime.utcnow()
    db.add(agent_run)    
    db.commit()

# == Agent Step == 
def create_agent_step(db: Session, step: AgentStep) -> AgentStep: 
    db.add(step)    
    db.commit()
    db.refresh(step)
    return step

# == Messages ==
def create_agent_message(db: Session, message: Message):
    db.add(message)    
    db.commit()

# == Logs ==
def insert_logs(db: Session, log: Log):
    db.add(log)    
    db.commit()

#== LLM Calls ==
def create_llm_call(db: Session, call: LLMCall): 
    call.created_at = datetime.utcnow()
    db.add(call)    
    db.commit()

# == Tool Call ==
def create_tool_call(db: Session, tool: ToolCall):
    db.add(tool)    
    db.commit()


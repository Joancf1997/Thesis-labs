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
from sqlalchemy.dialects.postgresql import UUID


#  == Session == 
def start_user_session(db: Session, user_id: UUID) -> UUID:
    s = UserSession(user_id=user_id)
    db.add(s)
    db.commit()
    db.refresh(s)
    return s.id

#  == Agent Run == 
def create_agent_run(db: Session, run: AgentRun) -> UUID:
    agent_run = AgentRun(
        session_id = run.session_id,
        user_input = run.user_input,
        status = run.status
    )
    db.add(agent_run)    
    db.commit()
    db.refresh(agent_run)
    return agent_run.id

def end_agent_run(db: Session): 
    agent_run = AgentRun(
        status = "completed",
        ended_at = datetime.utcnow()
    )
    db.add(agent_run)    
    db.commit()

# == Agent Step == 
def create_agent_step(db: Session, step: AgentStep) -> AgentStep: 
    step = AgentStep(
        run_id = step.run_id,
        step_order = step.step_order,
        step_type = step.step_type,
        input = step.input,
        output = step.output,
        meta = step.meta,
    )
    db.add(step)    
    db.commit()
    db.refresh(step)
    return step

# == Messages ==
def create_agent_message(db: Session, message: Message):
    message = Message(
        run_id= message.run_id, 
        role = message.role, 
        content = message.content, 
        message_type= message.message_type
    )
    db.add(message)    
    db.commit()

# == Logs ==
def insert_logs(db: Session, log: Log):
    log = Log(
        run_id = log.run_id,
        step_id = log.step_id,
        level = log.level,
        tool_call_id = log.tool_call_id,
        message = log.message,
        data = log.data
    )
    db.add(log)    
    db.commit()

#== LLM Calls ==
def create_llm_call(db: Session, call: LLMCall): 
    call = LLMCall(
        step_id = call.step_id,
        prompt =  call.prompt,
        response = call.response,
        model_name = call.model_name,
        temperature = call.temperature,
        token_usage = call.token_usage,
        created_at = datetime.utcnow()
    )
    db.add(call)    
    db.commit()




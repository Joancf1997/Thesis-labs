# app/models/agentStep.py
import uuid
from datetime import datetime
from sqlalchemy import Column, DateTime, ForeignKey, String, JSON
from sqlalchemy.orm import relationship
from app.db.base import Base
from app.models._types import GUID

class AgentStep(Base):
    __tablename__ = "agent_steps"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    run_id = Column(GUID(), ForeignKey("agent_runs.id"))
    step_type = Column(String)
    step_order = Column(String, nullable=True)
    input = Column(JSON)
    output = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    meta = Column(JSON)
    run = relationship("AgentRun", back_populates="steps")
    llm_calls = relationship("LLMCall", back_populates="step")
    tool_calls = relationship("ToolCall", back_populates="step")

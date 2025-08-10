import uuid
from app.db.base import Base
from datetime import datetime
from sqlalchemy.orm import relationship
from sqlalchemy import Column, DateTime, Text, ForeignKey, String, JSON
from app.models._types import GUID


class LLMCall(Base):
    __tablename__ = "llm_calls"
    
    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    step_id = Column(GUID(), ForeignKey("agent_steps.id"))
    prompt = Column(Text)
    response = Column(Text)
    model_name = Column(String)
    temperature = Column(String, nullable=True)
    token_usage = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    step = relationship("AgentStep", back_populates="llm_calls")
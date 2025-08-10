import uuid
from app.db.base import Base
from datetime import datetime
from sqlalchemy.orm import relationship
from sqlalchemy import Column, DateTime, Text, ForeignKey, String, JSON
from app.models._types import GUID


class ToolCall(Base):
    __tablename__ = "tool_calls"
    
    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    step_id = Column(GUID(), ForeignKey("agent_steps.id"))
    tool_id = Column(GUID(), nullable=True)
    input = Column(JSON)
    output = Column(JSON)
    status = Column(String)
    error_message = Column(Text)
    started_at = Column(DateTime, default=datetime.utcnow)
    ended_at = Column(DateTime, default=datetime.utcnow)
    meta = Column(JSON)
    step = relationship("AgentStep", back_populates="tool_calls")
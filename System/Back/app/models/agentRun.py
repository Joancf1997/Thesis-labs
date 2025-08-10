# app/models/agentRun.py
import uuid
from datetime import datetime
from sqlalchemy import Column, DateTime, ForeignKey, String, Text
from sqlalchemy.orm import relationship
from app.db.base import Base
from app.models._types import GUID

class AgentRun(Base):
    __tablename__ = "agent_runs"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    session_id = Column(GUID(), ForeignKey("user_sessions.id"))
    user_input = Column(Text)
    status = Column(String)
    started_at = Column(DateTime, default=datetime.utcnow)
    ended_at = Column(DateTime, nullable=True)

    session = relationship("UserSession", back_populates="runs")
    steps = relationship("AgentStep", back_populates="run")
    messages = relationship("Message", back_populates="run")

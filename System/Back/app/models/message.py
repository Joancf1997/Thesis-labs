# app/models/message.py
import uuid
from datetime import datetime
from sqlalchemy import Column, String, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from app.db.base import Base
from app.models._types import GUID

class Message(Base):
    __tablename__ = "messages"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    run_id = Column(GUID(), ForeignKey("agent_runs.id"), nullable=False, index=True)
    role = Column(String, nullable=False)          # e.g., "user" / "assistant"
    content = Column(String, nullable=False)
    message_type = Column(String, default="text")
    created_at = Column(DateTime, default=datetime.utcnow)
    run = relationship("AgentRun", back_populates="messages")

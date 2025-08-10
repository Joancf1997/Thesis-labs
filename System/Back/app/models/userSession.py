# app/models/userSession.py
import uuid
from datetime import datetime
from sqlalchemy import Column, DateTime
from sqlalchemy.orm import relationship
from app.db.base import Base
from app.models._types import GUID

class UserSession(Base):
    __tablename__ = "user_sessions"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    user_id = Column(GUID(), nullable=True)
    started_at = Column(DateTime, default=datetime.utcnow)
    ended_at = Column(DateTime, nullable=True)

    runs = relationship("AgentRun", back_populates="session")

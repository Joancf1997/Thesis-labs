import uuid
from app.db.base import Base
from datetime import datetime
from sqlalchemy import Column, DateTime, Text, ForeignKey, String, JSON
from app.models._types import GUID


class Log(Base):
    __tablename__ = "logs"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    run_id = Column(GUID(), ForeignKey("agent_runs.id"))
    step_id = Column(GUID(), ForeignKey("agent_steps.id"), nullable=True)
    tool_call_id = Column(GUID(), ForeignKey("tool_calls.id"), nullable=True)
    level = Column(String)
    message = Column(Text)
    data = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
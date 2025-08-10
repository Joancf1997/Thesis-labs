# app/schemas/assistant.py

from uuid import UUID
from typing import Literal, Optional
from pydantic import BaseModel, ConfigDict  # v2 style

class SessionBase(BaseModel):
    user_id: UUID
    model_config = ConfigDict(from_attributes=True)

class SessionInDb(SessionBase):
    # inherits config from SessionBase
    pass

class AgentRunBase(BaseModel):
    session_id: UUID
    user_input: str
    status: Literal["running", "completed", "failed"] = "running"
    model_config = ConfigDict(from_attributes=True)

class MessageBase(BaseModel):
    run_id: UUID
    role: str
    content: str
    message_type: str
    model_config = ConfigDict(from_attributes=True)

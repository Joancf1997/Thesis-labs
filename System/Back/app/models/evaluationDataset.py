from sqlalchemy import Column, Integer, Text
from app.db.base import Base
from app.models._types import GUID

class EvaluationDataset(Base):
    __tablename__ = "evaluation_dataset"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_query = Column(Text, nullable=False)
    task_plan = Column(Text, nullable=False)
    tools_used = Column(Text, nullable=False)

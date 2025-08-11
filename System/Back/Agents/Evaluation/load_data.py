import json
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from app.models.evaluationDataset import EvaluationDataset

# === DB CONFIGURATION ===
DATABASE_URL = "postgresql://joseandres:@localhost/news_AI"

Base = declarative_base()

# === DB CONNECTION ===
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

# Create table if it doesn't exist
Base.metadata.create_all(engine)

# === LOAD JSON FILE ===
with open("../Data/evaluation.json", "r") as f:
    data = json.load(f)

# === INSERT DATA INTO TABLE ===
session = SessionLocal()
try:
    for item in data:
        record = EvaluationDataset(
            user_query=item["user_query"],
            task_plan=item["task_plan"],
            tools_used=json.dumps(item["tools_used"])
        )
        session.add(record)
    session.commit()
    print("Data inserted successfully!")
except Exception as e:
    session.rollback()
    print(f"Error: {e}")
finally:
    session.close()

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.schemas.user import UserCreate, SessionInDb
from app.db.session import SessionLocal
from app.crud.assistant import start_user_session

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/start_user_session", response_model=SessionInDb)
def start_new_user_session(user: UserCreate, db: Session = Depends(get_db)):
    return start_user_session(db, user)


@router.post("/user_message", response_model=SessionInDb)
def user_message(user: UserCreate, db: Session = Depends(get_db)):
    return start_user_session(db, user)
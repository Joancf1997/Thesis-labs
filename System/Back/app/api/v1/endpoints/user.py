from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.schemas.user import UserCreate, UserInDB
from app.db.session import SessionLocal
from app.crud.user import create_user

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/", response_model=UserInDB)
def create_new_user(user: UserCreate, db: Session = Depends(get_db)):
    return create_user(db, user)

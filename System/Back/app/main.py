from fastapi import FastAPI
from app.api.v1.endpoints import user
from app.api.v1.assistant import assistant
from app.core.config import settings

app = FastAPI(title=settings.PROJECT_NAME)


# Routes
app.include_router(assistant.router, prefix=settings.API_V1_STR + "/assistant", tags=["assistans"])
app.include_router(user.router, prefix=settings.API_V1_STR + "/users", tags=["users"])

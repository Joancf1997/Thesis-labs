import os
import uuid
import logging
from .Agent import Agent
from app.db.init import init_db
from app.db.session import SessionLocal
from utils.utils import load_config, Settings



def configure_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        log_dir = os.path.join(base_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(log_dir, "../logs/system.log"))
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)


class ChatAssistant():
    def __init__(self):
        configure_logging()
        base_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(base_dir, "../config/settings.yaml")
        settings = Settings(**load_config(config_path))
        init_db()
        self.db = SessionLocal()
        user_id = uuid.uuid4()
        self.agent = Agent(settings, db=self.db, user_id=user_id)

    def ask(self, question: str):
        return self.agent.ask(question)
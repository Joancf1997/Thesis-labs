import os
import io
import sys
import json
import inspect
import uuid
import logging
from datetime import datetime
from typing import Dict, List, Literal, Optional

from pydantic import Field
from PIL import Image as PILImage
from typing_extensions import TypedDict
from IPython.display import Image, display
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain.chat_models import init_chat_model
from langchain.chains import ConversationChain
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import utils.tools as ToolsLlm
from utils.utility import Settings, load_prompt


class Evaluation(TypedDict):
    choice: Literal["Yes", "No"] = Field(description="'Yes' indicates accurate planning and 'No' indicates inaccurateplanning")
    reason: str = Field(description="Your reason for your choice")

# === Agent Class ===
class PlanJudge():
    def __init__(self, settings: Settings):
        self.llm_config = settings
        self.config_prompt()
        self.config_llm()


    def validation_router(self, state: dict) -> str:
        if len(state.get("plan")) == 0:
            return "direct_response"
        return "run_plan" if state.get("validation") else "task_planning"

    def config_prompt(self):
        self.planning_evaluation_prompt = load_prompt('3.gpt_score')

    def config_llm(self):
        base_llm = init_chat_model(model=self.llm_config.model_name, model_provider=self.llm_config.provider)
        self.plan_structure_llm = base_llm.with_structured_output(Evaluation)

    def evaluate(self, question: str, plan: str):
        print("Initializing judge plan evaluation ")
        response_stage_prompt = PromptTemplate(
            input_variables=["query", "plan"], template=self.planning_evaluation_prompt
        )
        prompt_str = response_stage_prompt.format(
            query=question, plan=plan
        )
        response = self.plan_structure_llm.invoke(prompt_str)
        return response

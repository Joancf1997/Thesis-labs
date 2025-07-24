import os
import io
import sys
import inspect
from pydantic import Field
from PIL import Image as PILImage
from langchain.chains import LLMChain
from typing_extensions import TypedDict
from IPython.display import Image, display
from langgraph.graph import START, StateGraph
from langchain.chat_models import init_chat_model
from typing import Dict, List, Literal,  Optional
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage
from langchain_core.prompts import SystemMessagePromptTemplate, PromptTemplate, ChatPromptTemplate
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import utils.tools as ToolsLlm
from utils.utility import Settings, load_prompt
import logging
from langgraph.prebuilt import create_react_agent
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import init_chat_model
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType




logger = logging.getLogger(__name__)

class State(TypedDict):
    question: str
    plan: Dict
    validation: bool
    outputs: Dict
    response: str 

class ArgPair(TypedDict):
    key: str
    value: str

class TaskResponseFormatter(TypedDict):
    task: Literal["summary_statistics", "predict_population_linear", "calculator", "plot_population"] = Field(description="Name of the tool to run")
    id: str = Field(description="Unique identifier for the task")
    dep: List[str] = Field(description="List of task ids this task depends on")
    args: Optional[List[ArgPair]] = Field(description="Arguments as list of (key, value) pairs")

class Plan(TypedDict):
    """Always use this tool to structure your response to the user in the planning phase."""
    plan: List[TaskResponseFormatter] = Field(description="List of tasks to be execute")

class Agent(): 
    def __init__(self, settings: Settings):
        self.llm_config = settings.llm
        self.session_config = {"configurable": {"thread_id": "1"}}
        self.config_llm()

    def ask(self, question: str):
        result = self.agent_executor.run(question)
        print(result)

    def config_prompt(self):
        self.planning_prompt = load_prompt('planning_stage')
        self.response_prompt = load_prompt('response_stage')
        self.task_planning_promp = ChatPromptTemplate([("system", self.planning_prompt), ("user", "Question: {input}")])
    
    def config_llm(self):
        base_llm = init_chat_model(model=self.llm_config.model_name, model_provider=self.llm_config.provider)
        memory = ConversationBufferMemory(
            memory_key="chat_history",    
            return_messages=True
        )
        
        tools = [
            Tool(
                name="predict_population_linear",
                func=ToolsLlm.predict_population_linear,
                description="Use to forecast population linearly"
            ),
            Tool(
                name="calculator",
                func=ToolsLlm.calculator,
                description="Use to evaluate arithmetic expressions"
            ),
            Tool(
                name="plot_population",
                func=ToolsLlm.plot_population,
                description="Use to plot population data"
            ),
        ]


        self.agent_executor = initialize_agent(
            tools=tools,
            llm=base_llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # ← this works
            memory=memory,
            verbose=False
        )



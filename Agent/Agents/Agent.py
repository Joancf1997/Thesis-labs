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
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

from sqlalchemy import (
    create_engine, Column, String, Text, DateTime, ForeignKey, JSON
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import sessionmaker, declarative_base, relationship

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import utils.tools as ToolsLlm
from utils.utility import Settings, load_prompt

logger = logging.getLogger(__name__)

# === SQLAlchemy Setup ===
DATABASE_URL = "postgresql://joseandres:@localhost/news_AI"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# === Models ===
class UserSession(Base):
    __tablename__ = "user_sessions"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=True)
    started_at = Column(DateTime, default=datetime.utcnow)
    ended_at = Column(DateTime, nullable=True)
    runs = relationship("AgentRun", back_populates="session")

class AgentRun(Base):
    __tablename__ = "agent_runs"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("user_sessions.id"))
    user_input = Column(Text)
    status = Column(String)
    started_at = Column(DateTime, default=datetime.utcnow)
    ended_at = Column(DateTime, nullable=True)
    meta = Column(JSON)
    session = relationship("UserSession", back_populates="runs")
    steps = relationship("AgentStep", back_populates="run")
    messages = relationship("Message", back_populates="run")

class AgentStep(Base):
    __tablename__ = "agent_steps"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id = Column(UUID(as_uuid=True), ForeignKey("agent_runs.id"))
    step_type = Column(String)
    step_order = Column(String, nullable=True)
    input = Column(JSON)
    output = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    meta = Column(JSON)
    run = relationship("AgentRun", back_populates="steps")
    llm_calls = relationship("LLMCall", back_populates="step")
    tool_calls = relationship("ToolCall", back_populates="step")

class LLMCall(Base):
    __tablename__ = "llm_calls"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    step_id = Column(UUID(as_uuid=True), ForeignKey("agent_steps.id"))
    prompt = Column(Text)
    response = Column(Text)
    model_name = Column(String)
    temperature = Column(String, nullable=True)
    token_usage = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    step = relationship("AgentStep", back_populates="llm_calls")

class ToolCall(Base):
    __tablename__ = "tool_calls"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    step_id = Column(UUID(as_uuid=True), ForeignKey("agent_steps.id"))
    tool_id = Column(UUID(as_uuid=True), nullable=True)
    input = Column(JSON)
    output = Column(JSON)
    status = Column(String)
    error_message = Column(Text)
    started_at = Column(DateTime, default=datetime.utcnow)
    ended_at = Column(DateTime, default=datetime.utcnow)
    meta = Column(JSON)
    step = relationship("AgentStep", back_populates="tool_calls")

class Message(Base):
    __tablename__ = "messages"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id = Column(UUID(as_uuid=True), ForeignKey("agent_runs.id"))
    role = Column(String)
    content = Column(Text)
    message_type = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    run = relationship("AgentRun", back_populates="messages")

class Log(Base):
    __tablename__ = "logs"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id = Column(UUID(as_uuid=True), ForeignKey("agent_runs.id"))
    step_id = Column(UUID(as_uuid=True), ForeignKey("agent_steps.id"), nullable=True)
    tool_call_id = Column(UUID(as_uuid=True), ForeignKey("tool_calls.id"), nullable=True)
    level = Column(String)
    message = Column(Text)
    data = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

# === Create tables ===
Base.metadata.drop_all(bind=engine)
Base.metadata.create_all(bind=engine)

# === TypedDicts for state ===
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
    plan: List[TaskResponseFormatter] = Field(description="List of tasks to be execute")

# === Agent Class ===
class Agent():
    def __init__(self, settings: Settings):
        self.llm_config = settings.llm
        self.session_config = {"configurable": {"thread_id": "1"}}
        self.state: State = {"question": "", "plan": [], "outputs": [], "response": ""}

        self.db = SessionLocal()
        self.session = UserSession()
        self.db.add(self.session)
        self.db.commit()
        self.db.refresh(self.session)
        self.session_id = self.session.id

        self.config_prompt()
        self.config_llm()
        self.workflow_config()

    def workflow_config(self):
        graph_builder = StateGraph(State)
        graph_builder.add_node("task_planning", self.task_planning)
        graph_builder.add_node("validate_plan", self.validate_plan)
        graph_builder.add_node("run_plan", self.run_plan)
        graph_builder.add_node("generate_response", self.generate_response)
        graph_builder.add_node("direct_response", self.direct_response)

        graph_builder.set_entry_point("task_planning")
        graph_builder.add_edge("task_planning", "validate_plan")
        graph_builder.add_conditional_edges("validate_plan", self.validation_router)
        graph_builder.add_edge("run_plan", "generate_response")
        graph_builder.set_finish_point("generate_response")
        graph_builder.set_finish_point("direct_response")

        memory = MemorySaver()
        self.request = graph_builder.compile(checkpointer=memory)

        image_bytes = self.request.get_graph().draw_mermaid_png()
        img = PILImage.open(io.BytesIO(image_bytes))
        img.save("graph_output.pdf", "PDF")

    def validation_router(self, state: dict) -> str:
        if len(state.get("plan")) == 0:
            return "direct_response"
        return "run_plan" if state.get("validation") else "task_planning"

    def ask(self, question: str):
        run = AgentRun(session_id=self.session_id, user_input=question, status="running")
        self.db.add(run)
        self.db.commit()
        self.db.refresh(run)
        self.run_id = run.id

        self.db.add(Message(run_id=self.run_id, role="user", content=question, message_type="plain"))
        self.db.commit()

        for step in self.request.stream({"question": question}, self.session_config, stream_mode="updates"):
            self.db.add(Log(run_id=self.run_id, level="info", message="Step update", data=step))
            self.db.commit()

        run.status = "completed"
        run.ended_at = datetime.utcnow()
        self.db.commit()

    def shutdown(self):
        self.session.ended_at = datetime.utcnow()
        self.db.commit()
        self.db.close()

    def config_prompt(self):
        self.planning_prompt = load_prompt('planning_stage')
        self.response_prompt = load_prompt('response_stage')
        self.direct_response_prompt = load_prompt('direct_response')
        self.task_planning_promp = ChatPromptTemplate([("system", self.planning_prompt), ("user", "Question: {input}")])

    def config_llm(self):
        base_llm = init_chat_model(model=self.llm_config.model_name, model_provider=self.llm_config.provider)
        memory = ConversationBufferMemory(return_messages=True)
        self.llm = ConversationChain(llm=base_llm, memory=memory, verbose=False)
        self.plan_structure_llm = base_llm.with_structured_output(Plan)

    def direct_response(self, state: State):
        step = AgentStep(run_id=self.run_id, step_type="direct_response", input=state["question"])
        self.db.add(step)
        self.db.commit()
        self.db.refresh(step)

        direct_prompt = PromptTemplate(input_variables=["user_request"], template=self.direct_response_prompt)
        prompt_str = direct_prompt.format(user_request=state["question"])
        response = self.llm.predict(input=prompt_str)

        self.db.add(LLMCall(step_id=step.id, prompt=prompt_str, response=response, model_name=self.llm_config.model_name))
        self.db.add(Message(run_id=self.run_id, role="assistant", content=response, message_type="plain"))
        step.output = response
        self.db.add(Log(run_id=self.run_id, step_id=step.id, level="info", message="Direct response generated",
                        data={"prompt": prompt_str, "response": response}))
        self.db.commit()
        return {"response": response}

    def task_planning(self, state: State):
        step = AgentStep(run_id=self.run_id, step_type="task_planning", input=state["question"])
        self.db.add(step)
        self.db.commit()
        self.db.refresh(step)

        prompt = self.task_planning_promp.invoke({"input": state["question"]})
        result = self.plan_structure_llm.invoke(prompt)

        self.db.add(LLMCall(step_id=step.id, prompt=str(prompt), response=str(result), model_name=self.llm_config.model_name))
        step.output = result
        self.db.commit()
        return {"plan": result["plan"]}

    def validate_plan(self, state: State):
        step = AgentStep(run_id=self.run_id, step_type="validate_plan", input=state["plan"])
        self.db.add(step)
        self.db.commit()
        self.db.refresh(step)

        task_ids = set()
        errors = []

        for task in state["plan"]:
            task_id = task.get('id')
            task_name = task.get('task')
            deps = task.get('dep', [])
            args = task.get('args', [])
            if not isinstance(task_id, str):
                errors.append(f"Task ID must be a string: {task_id}")
            elif task_id in task_ids:
                errors.append(f"Duplicate task ID found: {task_id}")
            else:
                task_ids.add(task_id)
            if task_name not in ToolsLlm.TASK_FUNCS:
                errors.append(f"Invalid task name: {task_name}")
            if not isinstance(deps, list):
                errors.append(f"Dependencies must be a list for task {task_id}")

        if errors:
            self.db.add(Log(run_id=self.run_id, step_id=step.id, level="error", message="Plan validation failed", data=errors))
            step.output = {"validation": False, "errors": errors}
            self.db.commit()
            return {"validation": False}

        self.db.add(Log(run_id=self.run_id, step_id=step.id, level="info", message="Plan validation passed", data=state["plan"]))
        step.output = {"validation": True}
        self.db.commit()
        return {"validation": True}

    def run_plan(self, state: State):
        outputs = {}
        remaining = state["plan"].copy()

        def resolve_args(args_list):
            resolved = {}
            for arg in args_list:
                k = arg['key']
                v = arg['value']
                if isinstance(v, str) and v.startswith("DEP_"):
                    dep_task_id = v[4:]
                    resolved[k] = outputs[dep_task_id]
                else:
                    resolved[k] = v
            return resolved

        while remaining:
            progress = False
            for task in remaining[:]:
                if all(dep in outputs for dep in task['dep']):
                    args = resolve_args(task.get('args', []))
                    step = AgentStep(run_id=self.run_id, step_type="run_plan", input=task)
                    self.db.add(step)
                    self.db.commit()
                    self.db.refresh(step)

                    func = ToolsLlm.TASK_FUNCS.get(task['task'])
                    result = func(**args)
                    self.db.add(ToolCall(step_id=step.id, input=task.get('args', []), output=result, status="success"))
                    outputs[task['id']] = result
                    remaining.remove(task)
                    progress = True
            if not progress:
                raise RuntimeError("Circular dependency or missing dependencies detected")
        return {"outputs": outputs}

    def generate_response(self, state: State):
        step = AgentStep(run_id=self.run_id, step_type="generate_response", input=state)
        self.db.add(step)
        self.db.commit()
        self.db.refresh(step)

        response_stage_prompt = PromptTemplate(
            input_variables=["user_request", "plan", "task_results"], template=self.response_prompt
        )
        prompt_str = response_stage_prompt.format(
            user_request=state["question"], plan=state["plan"], task_results=state["outputs"]
        )
        response = self.llm.predict(input=prompt_str)

        self.db.add(LLMCall(step_id=step.id, prompt=prompt_str, response=response, model_name=self.llm_config.model_name))
        self.db.add(Message(run_id=self.run_id, role="assistant", content=response, message_type="plain"))
        step.output = response
        self.db.commit()
        return {"response": response}

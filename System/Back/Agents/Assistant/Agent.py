import io
import uuid
import logging
from pydantic import Field
from datetime import datetime
from PIL import Image as PILImage
from sqlalchemy.orm import Session
from langgraph.graph import StateGraph
from typing_extensions import TypedDict
from langchain.chains import ConversationChain
from typing import Dict, List, Literal, Optional
from langchain.chat_models import init_chat_model
from langchain.memory import ConversationBufferMemory
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
import utils.tools as ToolsLlm
from utils.utils import Settings, load_prompt
logger = logging.getLogger(__name__)
from app.models.log import Log
from app.models.agentRun import AgentRun
from app.models.message import Message
from app.models.agentStep import AgentStep
from app.models.llmCall import LLMCall
from app.models.toolCall import ToolCall
from app.crud.assistant import start_user_session, create_agent_run, create_agent_message, insert_logs, end_agent_run, create_agent_step, create_llm_call, create_tool_call

# === TypedDicts for state ===
class State(TypedDict):
    question: str
    plan: Dict
    validation: bool
    outputs: Dict
    response: str

# === Structure response == 
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
    def __init__(self, settings: Settings, db: Session, user_id: uuid):
        self.db = db
        self.llm_config = settings.llm
        self.session_config = {"configurable": {"thread_id": "1"}}
        self.state: State = {"question": "", "plan": [], "validation": False, "outputs": [], "response": ""}
        self.session_id = start_user_session(db, user_id)
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
        self.run = create_agent_run(self.db, AgentRun(session_id=self.session_id, user_input=question))
        create_agent_message(self.db, Message(run_id=self.run.id, role="user", content=question, message_type="plain"))

        for step in self.request.stream({"question": question}, self.session_config, stream_mode="updates"):
            insert_logs(self.db, Log(run_id=self.run.id, level="info", message="Step update", data=step))

        end_agent_run(self.db, self.run)

    def shutdown(self):
        self.session.ended_at = datetime.utcnow()
        self.db.commit()
        self.db.close()

    def config_prompt(self):
        self.planning_prompt = load_prompt('1.planning_stage')
        self.response_prompt = load_prompt('2.response_stage')
        self.direct_response_prompt = load_prompt('2.direct_response')
        self.task_planning_promp = ChatPromptTemplate([("system", self.planning_prompt), ("user", "Question: {input}")])

    def config_llm(self):
        base_llm = init_chat_model(model=self.llm_config.model_name, model_provider=self.llm_config.provider)
        memory = ConversationBufferMemory(return_messages=True)
        self.llm = ConversationChain(llm=base_llm, memory=memory, verbose=False)
        self.plan_structure_llm = base_llm.with_structured_output(Plan)

    def direct_response(self, state: State):
        step = create_agent_step(self.db, AgentStep(run_id=self.run.id, step_type="direct_response", input=state["question"]))
        direct_prompt = PromptTemplate(input_variables=["user_request"], template=self.direct_response_prompt)
        prompt_str = direct_prompt.format(user_request=state["question"])
        response = self.llm.predict(input=prompt_str)

        create_llm_call(self.db, LLMCall(step_id=step.id, prompt=prompt_str, response=response, model_name=self.llm_config.model_name))
        create_agent_message(self.db, Message(run_id=self.run.id, role="assistant", content=response, message_type="plain"))
        insert_logs(self.db, Log(run_id=self.run.id, step_id=step.id, level="info", message="Direct response generated", data={"prompt": prompt_str, "response": response}))
        return {"response": response}
    
    def check_tools(self, pred, label):
        items_metric = len(set(pred).intersection(label))     # items
        order_metric = sum(1 for i in range(min(len(pred), len(label))) if pred[i] == label[i])  # Order
        print("items", items_metric, "order", order_metric, "total", len(label))
        return {"items": items_metric, "order": order_metric, "total": len(label)}

    def task_planning(self, state: State):
        step = create_agent_step(self.db, AgentStep(run_id=self.run.id, step_type="task_planning", input=state["question"]))
        prompt = self.task_planning_promp.invoke({"input": state["question"]})
        result = self.plan_structure_llm.invoke(prompt)

        if True: 
            task_list = [item["task"] for item in result["plan"]]
            print(task_list)
            self.check_tools(task_list, ['predict_population_linear', 'plot_population'])  

        create_llm_call(self.db, LLMCall(step_id=step.id, prompt=str(prompt), response=str(result), model_name=self.llm_config.model_name))
        return {"plan": result["plan"]}

    def validate_plan(self, state: State):
        step = create_agent_step(self.db, AgentStep(run_id=self.run.id, step_type="validate_plan", input=state["plan"]))
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
            insert_logs(self.db, Log(run_id=self.run.id, step_id=step.id, level="error", message="Plan validation failed", data=errors))
            return {"validation": False}

        insert_logs(self.db, Log(run_id=self.run.id, step_id=step.id, level="info", message="Plan validation passed", data=state["plan"]))
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
                    step = create_agent_step(self.db, AgentStep(run_id=self.run.id, step_type="run_plan", input=task))
                    func = ToolsLlm.TASK_FUNCS.get(task['task'])
                    result = func(**args)
                    create_tool_call(self.db, ToolCall(step_id=step.id, input=task.get('args', []), output=result, status="success"))
                    outputs[task['id']] = result
                    remaining.remove(task)
                    progress = True
            if not progress:
                raise RuntimeError("Circular dependency or missing dependencies detected")
        return {"outputs": outputs}

    def generate_response(self, state: State):
        step = create_agent_step(self.db, AgentStep(run_id=self.run.id, step_type="generate_response", input=state))

        response_stage_prompt = PromptTemplate(
            input_variables=["user_request", "plan", "task_results"], template=self.response_prompt
        )
        prompt_str = response_stage_prompt.format(
            user_request=state["question"], plan=state["plan"], task_results=state["outputs"]
        )
        response = self.llm.predict(input=prompt_str)
        
        create_llm_call(self.db, LLMCall(step_id=step.id, prompt=prompt_str, response=response, model_name=self.llm_config.model_name))
        print(response)
        return {"response": response}

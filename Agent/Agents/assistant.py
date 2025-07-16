import os
import io
import sys
import inspect
from pydantic import Field
from PIL import Image as PILImage
from typing_extensions import TypedDict
from IPython.display import Image, display
from langgraph.graph import START, StateGraph
from langchain.chat_models import init_chat_model
from typing import Dict, List, Literal,  Optional
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.prompts import ChatPromptTemplate
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.utility import load_config, Settings, load_prompt
import utils.tools as ToolsLlm


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
        self.state: State = {"question":"", "plan": [], "outputs": [], "response": ""}
        self.config_prompt()
        self.config_llm()
        graph_builder = StateGraph(State).add_sequence(
            [self.task_planning, self.validate_plan, self.run_plan, self.generate_response]
        )
        graph_builder.add_edge(START, "task_planning")
        memory = MemorySaver()
        self.request = graph_builder.compile(checkpointer=memory, interrupt_before=["run_plan"])
        image_bytes = self.request.get_graph().draw_mermaid_png()
        img = PILImage.open(io.BytesIO(image_bytes))
        img.save("graph_output.pdf", "PDF")

    def config_prompt(self):
        self.planning_prompt = load_prompt("planning_stage")
        self.response_prompt = load_prompt("response_stage")
        self.planning_prompt_temp = ChatPromptTemplate([("system", self.planning_prompt), ("user", "Question: {input}")])
    
    def config_llm(self):
        self.llm = init_chat_model(model=self.llm_config.model_name, model_provider=self.llm_config.provider)
        self.plan_structure_llm = self.llm.with_structured_output(Plan)
    
    def task_planning(self, state: State): 
      prompt = self.planning_prompt_temp.invoke({"input": state["question"],})
      result = self.plan_structure_llm.invoke(prompt)
      return {"plan": result["plan"]}

    def validate_plan(self, state: State):
        task_ids = set()
        errors = []
        for task in state["plan"]:
            task_id = task.get('id')
            task_name = task.get('task')
            deps = task.get('dep', [])
            args = task.get('args', [])

            # Check ID
            if not isinstance(task_id, str):
                errors.append(f"Task ID must be a string: {task_id}")
            elif task_id in task_ids:
                errors.append(f"Duplicate task ID found: {task_id}")
            else:
                task_ids.add(task_id)

            # Check task name
            if task_name not in ToolsLlm.TASK_FUNCS:
                errors.append(f"Invalid task name: {task_name}")

            # Check deps
            if not isinstance(deps, list):
                errors.append(f"Dependencies must be a list for task {task_id}")
            else:
                for dep_id in deps:
                    if not isinstance(dep_id, str):
                        errors.append(f"Dependency ID must be string in task {task_id}: {dep_id}")

            # Check args format
            if not isinstance(args, list):
                errors.append(f"Args must be a list for task {task_id}")
            else:
                for arg in args:
                    if not isinstance(arg, dict) or 'key' not in arg or 'value' not in arg:
                        errors.append(f"Invalid arg format in task {task_id}: {arg}")

        # validate if dependencies exist
        for task in state["plan"]:
            for dep_id in task.get('dep', []):
                if dep_id not in task_ids:
                    errors.append(f"Task {task['id']} has unknown dependency: {dep_id}")

            for arg in task.get('args', []):
                val = arg['value']
                if isinstance(val, str) and val.startswith("DEP_"):
                    dep_ref = val[4:]
                    if dep_ref not in task_ids:
                        errors.append(f"Task {task['id']} references unknown DEP value: {val}")

        # Validate args against function signature
        for task in state["plan"]:
            func = ToolsLlm.TASK_FUNCS.get(task['task'])
            if func:
                sig = inspect.signature(func)
                expected_args = set(sig.parameters.keys())
                actual_args = set(arg['key'] for arg in task['args'])
                if not actual_args.issubset(expected_args):
                    extra = actual_args - expected_args
                    errors.append(f"Task {task['id']} has unexpected args: {extra}")
                missing = expected_args - actual_args
                if missing:
                    errors.append(f"Task {task['id']} is missing args: {missing}")
        if errors:
            raise ValueError("Plan validation failed:\n" + "\n".join(errors))
        return {"validation": True}

    def run_plan(self, state: State): 
        outputs = {}
        def resolve_args(args_list):
            resolved = {}
            for arg in args_list:
                k = arg['key']
                v = arg['value']
                if isinstance(v, str) and v.startswith("DEP_"):
                    dep_task_id = v[4:]
                    if dep_task_id not in outputs:
                        raise ValueError(f"Dependency output for {dep_task_id} not ready")
                    resolved[k] = outputs[dep_task_id]
                else:
                    resolved[k] = v
            return resolved
        remaining = state["plan"].copy()
        while remaining:
            progress = False
            for task in remaining[:]: 
                if all(dep in outputs for dep in task['dep']):
                    args_list = task.get('args', []) 
                    args = resolve_args(args_list)

                    func = ToolsLlm.TASK_FUNCS.get(task['task'])
                    if not func:
                        raise ValueError(f"No function defined for task {task['task']}")
                    result = func(**args)

                    outputs[task['id']] = result
                    remaining.remove(task)
                    progress = True
            if not progress:
                raise RuntimeError("Circular dependency or missing dependencies detected")
        return {"outputs":  outputs}

    def generate_response(self, state): 
        return {"response": "Respuesta del agente"}


# Agent Instance
settings = Settings(**load_config("config/settings.yaml"))
agent = Agent(settings)


# Session thread
config = {"configurable": {"thread_id": "1"}}
for step in agent.request.stream(
    {"question": "Forecast the population for 2029."}, 
    config, 
    stream_mode="updates"
):
    print(step)
try:
    user_approval = input("Do you want to go to execute query? (yes/no): ")
except Exception:
    user_approval = "no"

if user_approval.lower() == "yes":
    for step in agent.request.stream(None, config, stream_mode="updates"):
        print(step)
else:
    print("Operation cancelled by user.")


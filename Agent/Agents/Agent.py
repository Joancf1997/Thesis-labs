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
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory


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
        self.state: State = {"question":"", "plan": [], "outputs": [], "response": ""}
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
        # self.request = graph_builder.compile(checkpointer=memory, interrupt_before=["run_plan"])
        self.request = graph_builder.compile(checkpointer=memory)

        image_bytes = self.request.get_graph().draw_mermaid_png()
        img = PILImage.open(io.BytesIO(image_bytes))
        img.save("graph_output.pdf", "PDF")

    def validation_router(self, state: dict) -> str:
        logging.info("validation router..")
        if len(state.get("plan")) == 0:
            logging.info("No plan found, skipping validation.")
            return "direct_response"
        
        if state.get("validation"):
            return "run_plan"
        else:
            return "task_planning"
        
    def ask(self, question: str):
        if False:
            for step in self.request.stream(
                {"question": question}, 
                self.session_config, 
                stream_mode="updates"
            ):
                print(step)
        else: 
            result = self.request.invoke(
                {"question": question},
                config=self.session_config
            )
            print(result['response'])


    def config_prompt(self):
        self.planning_prompt = load_prompt('planning_stage')
        self.response_prompt = load_prompt('response_stage')
        self.direct_response_prompt = load_prompt('direct_response')
        self.task_planning_promp = ChatPromptTemplate([("system", self.planning_prompt), ("user", "Question: {input}")])
    
    def config_llm(self):
        base_llm = init_chat_model(model=self.llm_config.model_name, model_provider=self.llm_config.provider)
        memory = ConversationBufferMemory(return_messages=True)
        self.llm = ConversationChain(
            llm=base_llm,
            memory=memory,
            verbose=False
        )
        
        self.plan_structure_llm = base_llm.with_structured_output(Plan)
    
    def direct_response(self, state: State):
        logging.info("Direct response stage started...")
        direct_prompt = PromptTemplate(
            input_variables=["user_request"],
            template=self.direct_response_prompt,
        )
        prompt_str = direct_prompt.format(user_request=state["question"])
        response = self.llm.predict(input=prompt_str)
        logging.info("Model response: %s", response)
        return {"response": response}


    def task_planning(self, state: State): 
      logging.info("Task planning stage started...")
      prompt = self.task_planning_promp.invoke({"input": state["question"],})
      result = self.plan_structure_llm.invoke(prompt)
      logging.info("Plan: %s", result["plan"])
      return {"plan": result["plan"]}

    def validate_plan(self, state: State):
        logging.info("Validate planning stage started...")
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
            logging.info("Plan validation failed:\n" + "\n".join(errors))
            # raise ValueError("Plan validation failed:\n" + "\n".join(errors))
            return {"validation": False}
        
        logging.info("Plan validation PASSED...")
        return {"validation": True}

    def run_plan(self, state: State): 
        logging.info("Run plan stage started...")
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

    def generate_response(self, state: State): 
        logging.info("Response stage started...")
        response_stage_prompt = PromptTemplate(
            input_variables=["user_request", "plan", "task_results"],
            template=self.response_prompt,
        )
        prompt_str = response_stage_prompt.format(
            user_request=state["question"],
            plan=state["plan"],
            task_results=state["outputs"]
        )
        response = self.llm.predict(input=prompt_str)
        logging.info("Model response: %s", response)
        return {"response": response}



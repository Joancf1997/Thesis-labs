import os
import json
import uuid
from sqlalchemy import text
from app.db.init import init_db
from Assistant.Agent import Agent
from app.db.session import SessionLocal
from utils.utils import load_config, Settings
from app.models.evaluationDataset import EvaluationDataset
from app.models.experiments import (
    Experiment, ExperimentRun,
    PlanningGpt4Eval, TaskEval, UserFeedback, InterpretabilityRating
)
from app.crud.evaluation import start_agent_evaluation, create_experiment_run, create_task_evaluation


class AgentEval():
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(base_dir, "../config/settings.yaml")
        settings = Settings(**load_config(config_path))
        init_db()                  
        self.db = SessionLocal()   
        user_id = uuid.uuid4()
        self.agent = Agent(settings, db=self.db, user_id=user_id)
        self.dataset = self.load_task_plan_dataset()


    def load_task_plan_dataset(self):
        dataset = []
        rows = self.db.query(EvaluationDataset).order_by(EvaluationDataset.id).all()
        for r in rows:
            tools = r.tools_used
            try:
                tools = json.loads(tools)
            except json.JSONDecodeError:
                pass
            dataset.append({
                "id": r.id,
                "user_query": r.user_query,
                "task_plan": r.task_plan,
                "tools_used": tools,
            })
        return dataset
    
    def check_tools(self, pred, label):
        pred_set = set(pred)
        label_set = set(label)
        tp = len(pred_set & label_set)
        fp = len(pred_set - label_set)
        fn = len(label_set - pred_set)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    
    def plan_task_eval(self, pred, labels):
        return self.check_tools(pred, labels)
    
    def eval(self, name: str, description: str):
        # Create the experiment 
        exp = start_agent_evaluation(self.db, Experiment(name=name, description=description))
        
        # Dataset Iteration 
        for item in self.dataset: 
            agent_run, tools = self.agent.ask(item['user_query'])
            run_id = agent_run.id

            # Experiment run 
            create_experiment_run(self.db, ExperimentRun(experiment_id=exp.id, run_id=run_id, dataset_id=item['id']))

            # GPT Judge Eval 


            # Heuristic Task Eval
            task_list = [item["task"] for item in tools['plan']]
            task_evaluation = self.plan_task_eval(task_list, item['tools_used']) 
            create_task_evaluation(self.db,TaskEval(run_id=run_id, 
                                   gold_task_sequence=task_list, 
                                   predicted_task_sequence=item['tools_used'], 
                                   tp=task_evaluation['tp'],
                                   fp=task_evaluation['fp'],
                                   fn=task_evaluation['fn'],
                                   f1=task_evaluation['f1'],
                                   precision=task_evaluation['precision'],
                                   recall=task_evaluation['recall']
                                ) ) 

            

            

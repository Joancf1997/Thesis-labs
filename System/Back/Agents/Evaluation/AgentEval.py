import os
import json
import uuid
from sqlalchemy import text
from app.db.init import init_db
from Assistant.Agent import Agent
from app.db.session import SessionLocal
from utils.utils import load_config, Settings
from app.models.evaluationDataset import EvaluationDataset

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
                "user_query": r.user_query,
                "task_plan": r.task_plan,
                "tools_used": tools,
            })
        return dataset

    
    def check_tools(self, pred, label):
        items_metric = len(set(pred).intersection(label))     # items
        order_metric = sum(1 for i in range(min(len(pred), len(label))) if pred[i] == label[i])  # Order
        print("items", items_metric, "order", order_metric, "total", len(label))
        return {"items": items_metric, "order": order_metric, "total": len(label)}
    
    # def plan_task_eval(self):
    #     task_list = [item["task"] for item in result["plan"]]
    #     self.check_tools(task_list, ['predict_population_linear', 'plot_population'])  
    
    def eval(self):
        for item in self.dataset: 
            print(item['user_query'])
            print(item['tools_used'])
            self.agent.ask(item['user_query'])

import os
import sys 
import logging
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Agents.Agent import Agent
import json
from utils.utility import load_config, Settings



class AgentEval():
    def __init__(self):
        # create agent 
        base_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(base_dir, "../config/settings.yaml")
        settings = Settings(**load_config(config_path))
        self.agent = Agent(settings)
        self.dataset = self.load_task_plan_dataset("../Data/evaluation.json")

    def load_task_plan_dataset(self, json_file):
        with open(json_file, 'r') as file:
            dataset = json.load(file)
        return dataset
    
    def check_tools(self, pred, label):
        items_metric = len(set(pred).intersection(label))     # items
        order_metric = sum(1 for i in range(min(len(pred), len(label))) if pred[i] == label[i])  # Order
        print("items", items_metric, "order", order_metric, "total", len(label))
        return {"items": items_metric, "order": order_metric, "total": len(label)}
    
    def plan_task_eval(self):
        task_list = [item["task"] for item in result["plan"]]
        self.check_tools(task_list, ['predict_population_linear', 'plot_population'])  

    
    def eval(self):
        for item in self.dataset: 
            self.agent.ask(item['user_query'])

            # print(item['user_query'])
            # print(item['tools_used'])
import os
import yaml
from pydantic import BaseModel

def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)
    
def load_prompt(name: str) -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, f"../config/prompts/{name}.txt")
    with open(config_path, "r") as file:
        return file.read()
    
class LLMConfig(BaseModel):
    provider: str
    model_name: str
    temperature: float
    max_tokens: int

class PromptConfig(BaseModel):
    planning: str
    response: str

class Settings(BaseModel):
    llm: LLMConfig

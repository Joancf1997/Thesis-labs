import yaml
from pydantic import BaseModel

def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)
    
def load_prompt(name: str) -> str:
    with open(f"config/prompts/{name}.txt", "r") as file:
        return file.read()
    
class LLMConfig(BaseModel):
    provider: str
    model_name: str
    temperature: float
    max_tokens: int

class Settings(BaseModel):
    llm: LLMConfig

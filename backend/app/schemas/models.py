from pydantic import BaseModel

class ModelInfo(BaseModel):
    id: str
    name: str
    active: bool
    loaded: bool
    task_type: str
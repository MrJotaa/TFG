from pydantic import BaseModel

class ModelInfo(BaseModel):
    id: str
    name: str
    active: bool
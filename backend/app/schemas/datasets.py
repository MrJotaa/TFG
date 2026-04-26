from pydantic import BaseModel


class DatasetImage(BaseModel):
    id: str
    filename: str
    path: str


class DatasetResponse(BaseModel):
    id: str
    name: str
    owner_id: str
    image_count: int
    path: str
    images: list[DatasetImage]
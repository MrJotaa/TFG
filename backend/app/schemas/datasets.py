from pydantic import BaseModel


class DatasetImage(BaseModel):
    id: str
    filename: str
    url: str


class DatasetResponse(BaseModel):
    id: str
    name: str
    owner_id: str
    image_count: int
    images: list[DatasetImage]
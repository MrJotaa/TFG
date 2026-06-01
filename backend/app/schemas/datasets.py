from pydantic import BaseModel


class DatasetImage(BaseModel):
    id: str
    filename: str
    url: str | None = None
    label: str | None = None


class DatasetResponse(BaseModel):
    id: str
    name: str
    owner_id: str
    image_count: int
    images: list[DatasetImage]

class DatasetImageLabelUpdate(BaseModel):
    label: str | None = None
from pydantic import BaseModel


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class DetectionPrediction(BaseModel):
    label: str
    confidence: float
    bbox: BoundingBox


class DetectionResponse(BaseModel):
    model_id: str
    model_name: str
    detections: list[DetectionPrediction]
    annotated_image_url: str | None = None
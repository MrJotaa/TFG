from pydantic import BaseModel

from schemas.detection import DetectionPrediction


class GradCamRequest(BaseModel):
    dataset_id: str
    image_id: str


class GradCamResponse(BaseModel):
    result_id: str
    prediction: str
    confidence: float
    gradcam_image_url: str


class DetectionExplainRequest(BaseModel):
    dataset_id: str
    image_id: str


class DetectionExplainResponse(BaseModel):
    result_id: str
    model_id: str
    model_name: str
    explanation_type: str
    explanation_image_url: str
    detections: list[DetectionPrediction]
from pydantic import BaseModel


class GradCamRequest(BaseModel):
    dataset_id: str
    image_id: str


class GradCamResponse(BaseModel):
    prediction: str
    confidence: float
    gradcam_image: str
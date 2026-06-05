from fastapi import APIRouter

from schemas.detection import DetectionRequest, DetectionResponse
from services.detection_service import detect_objects

router = APIRouter()


@router.post("/detect", response_model=DetectionResponse)
async def detect_image(request: DetectionRequest) -> DetectionResponse:
    return await detect_objects(request)
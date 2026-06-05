from fastapi import APIRouter

from schemas.detection_evaluation import (
    DetectionEvaluationRequest,
    DetectionEvaluationResponse,
)
from services.detection_evaluation_service import evaluate_detection_dataset

router = APIRouter()


@router.post("/detect/evaluate", response_model=DetectionEvaluationResponse)
async def evaluate_detection(
    request: DetectionEvaluationRequest,
) -> DetectionEvaluationResponse:
    return await evaluate_detection_dataset(request)
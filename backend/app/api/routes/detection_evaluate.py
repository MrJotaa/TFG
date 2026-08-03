from fastapi import APIRouter, Depends, Query

from api.dependencies.auth import require_current_user
from schemas.detection_evaluation import (
    DetectionEvaluationHistoryItem,
    DetectionEvaluationRequest,
    DetectionEvaluationResponse,
)
from schemas.users import UserResponse
from services.detection_evaluation_persistence import (
    get_detection_evaluation,
    list_detection_evaluations,
)
from services.detection_evaluation_service import evaluate_detection_dataset

router = APIRouter()


@router.post("/detect/evaluate", response_model=DetectionEvaluationResponse)
async def evaluate_detection(
    request: DetectionEvaluationRequest,
) -> DetectionEvaluationResponse:
    return await evaluate_detection_dataset(request)


@router.get(
    "/detect/evaluations",
    response_model=list[DetectionEvaluationHistoryItem],
)
async def get_current_user_detection_evaluations(
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    current_user: UserResponse = Depends(require_current_user),
) -> list[DetectionEvaluationHistoryItem]:
    return await list_detection_evaluations(
        owner_id=current_user.id,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/detect/evaluations/{evaluation_id}",
    response_model=DetectionEvaluationResponse,
)
async def get_current_user_detection_evaluation(
    evaluation_id: str,
    current_user: UserResponse = Depends(require_current_user),
) -> DetectionEvaluationResponse:
    return await get_detection_evaluation(
        evaluation_id=evaluation_id,
        owner_id=current_user.id,
    )

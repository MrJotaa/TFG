from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse

from api.dependencies.auth import require_current_user
from schemas.detection_evaluation import (
    DetectionEvaluationEigenCamResponse,
    DetectionEvaluationHistoryItem,
    DetectionEvaluationRequest,
    DetectionEvaluationResponse,
)
from schemas.users import UserResponse
from services.detection_evaluation_explainability_service import (
    generate_detection_evaluation_image_eigencam,
)
from services.detection_evaluation_persistence import (
    get_detection_evaluation,
    list_detection_evaluations,
)
from services.detection_evaluation_service import evaluate_detection_dataset
from services.evaluation_progress import stream_evaluation

router = APIRouter()


@router.post("/detect/evaluate")
async def evaluate_detection(
    request: DetectionEvaluationRequest,
    current_user: UserResponse = Depends(require_current_user),
) -> DetectionEvaluationResponse:
    return await evaluate_detection_dataset(
        request,
        owner_id=current_user.id,
        retention_seconds=current_user.result_retention_seconds,
    )


@router.post("/detect/evaluate/stream")
async def evaluate_detection_stream(
    request: DetectionEvaluationRequest,
    current_user: UserResponse = Depends(require_current_user),
) -> StreamingResponse:
    async def runner(progress_callback, cancellation_token):
        return await evaluate_detection_dataset(
            request,
            owner_id=current_user.id,
            retention_seconds=current_user.result_retention_seconds,
            progress_callback=progress_callback,
            cancellation_token=cancellation_token,
        )

    return StreamingResponse(
        stream_evaluation(runner),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get(
    "/detect/evaluations",
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
)
async def get_current_user_detection_evaluation(
    evaluation_id: str,
    current_user: UserResponse = Depends(require_current_user),
) -> DetectionEvaluationResponse:
    return await get_detection_evaluation(
        evaluation_id=evaluation_id,
        owner_id=current_user.id,
    )


@router.post(
    "/detect/evaluations/{evaluation_id}/images/{image_id}/eigencam",
)
async def detection_evaluation_image_eigencam(
    evaluation_id: str,
    image_id: str,
    current_user: UserResponse = Depends(require_current_user),
) -> DetectionEvaluationEigenCamResponse:
    return await generate_detection_evaluation_image_eigencam(
        evaluation_id=evaluation_id,
        image_id=image_id,
        current_user_id=current_user.id,
    )

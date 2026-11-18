from fastapi import APIRouter, Depends, Request

from api.dependencies.auth import require_current_user
from schemas.explainability import (
    DetectionExplainRequest,
    DetectionExplainResponse,
    GradCamRequest,
    GradCamResponse,
)
from schemas.users import UserResponse
from services.explainability_service import (
    generate_detection_explainability,
    generate_gradcam_for_image,
)
from services.request_cancellation import run_cancellable_request

router = APIRouter()


@router.post("/explain/gradcam")
async def gradcam(
    request: GradCamRequest,
    http_request: Request,
    current_user: UserResponse = Depends(require_current_user),
) -> GradCamResponse:
    return await run_cancellable_request(
        http_request,
        lambda cancellation_token: generate_gradcam_for_image(
            dataset_id=request.dataset_id,
            image_id=request.image_id,
            owner_id=current_user.id,
            cancellation_token=cancellation_token,
        ),
    )


@router.post("/explain/detection")
async def explain_detection(
    request: DetectionExplainRequest,
    http_request: Request,
    current_user: UserResponse = Depends(require_current_user),
) -> DetectionExplainResponse:
    return await run_cancellable_request(
        http_request,
        lambda cancellation_token: generate_detection_explainability(
            request,
            owner_id=current_user.id,
            cancellation_token=cancellation_token,
        ),
    )

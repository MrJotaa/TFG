from fastapi import APIRouter, Request

from schemas.explainability import (
    DetectionExplainRequest,
    DetectionExplainResponse,
    GradCamRequest,
    GradCamResponse,
)
from services.explainability_service import (
    generate_detection_explainability,
    generate_gradcam_for_image,
)
from services.request_cancellation import run_cancellable_request

router = APIRouter()


@router.post("/explain/gradcam", response_model=GradCamResponse)
async def gradcam(
    request: GradCamRequest,
    http_request: Request,
) -> GradCamResponse:
    return await run_cancellable_request(
        http_request,
        lambda cancellation_token: generate_gradcam_for_image(
            dataset_id=request.dataset_id,
            image_id=request.image_id,
            cancellation_token=cancellation_token,
        ),
    )


@router.post("/explain/detection", response_model=DetectionExplainResponse)
async def explain_detection(
    request: DetectionExplainRequest,
    http_request: Request,
) -> DetectionExplainResponse:
    return await run_cancellable_request(
        http_request,
        lambda cancellation_token: generate_detection_explainability(
            request,
            cancellation_token=cancellation_token,
        ),
    )

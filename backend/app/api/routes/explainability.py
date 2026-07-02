from fastapi import APIRouter

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

router = APIRouter()


@router.post("/explain/gradcam", response_model=GradCamResponse)
async def gradcam(request: GradCamRequest) -> GradCamResponse:
    return await generate_gradcam_for_image(
        dataset_id=request.dataset_id,
        image_id=request.image_id,
    )


@router.post("/explain/detection", response_model=DetectionExplainResponse)
async def explain_detection(
    request: DetectionExplainRequest,
) -> DetectionExplainResponse:
    return await generate_detection_explainability(request)

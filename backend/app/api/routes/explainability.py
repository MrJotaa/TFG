from fastapi import APIRouter

from schemas.explainability import GradCamRequest, GradCamResponse
from services.explainability_service import generate_gradcam_for_image

router = APIRouter()


@router.post("/explain/gradcam", response_model=GradCamResponse)
async def gradcam(request: GradCamRequest) -> GradCamResponse:
    return await generate_gradcam_for_image(
        dataset_id=request.dataset_id,
        image_id=request.image_id,
    )
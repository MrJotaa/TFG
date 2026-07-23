from fastapi import APIRouter, Depends

from api.dependencies.auth import require_current_user
from schemas.evaluation import (
    EvaluationGradCamResponse,
    EvaluationRequest,
    EvaluationResponse,
)
from schemas.users import UserResponse
from services.evaluation_gradcam_service import (
    generate_evaluation_image_gradcam,
)
from services.evaluation_service import evaluate_dataset

router = APIRouter()


@router.post("/evaluate", response_model=EvaluationResponse)
async def evaluate(request: EvaluationRequest) -> EvaluationResponse:
    return await evaluate_dataset(request)


@router.post(
    "/evaluations/{evaluation_id}/images/{image_id}/gradcam",
    response_model=EvaluationGradCamResponse,
)
async def evaluation_image_gradcam(
    evaluation_id: str,
    image_id: str,
    current_user: UserResponse = Depends(require_current_user),
) -> EvaluationGradCamResponse:
    return await generate_evaluation_image_gradcam(
        evaluation_id=evaluation_id,
        image_id=image_id,
        current_user_id=current_user.id,
    )

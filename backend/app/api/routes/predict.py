from fastapi import APIRouter, Depends, Request

from api.dependencies.auth import require_current_user
from schemas.predict import PredictRequest, PredictResponse
from schemas.users import UserResponse
from services.inference_service import predict_image
from services.request_cancellation import run_cancellable_request

router = APIRouter()


@router.post("/predict")
async def predict(
    request: PredictRequest,
    http_request: Request,
    current_user: UserResponse = Depends(require_current_user),
) -> PredictResponse:
    return await run_cancellable_request(
        http_request,
        lambda cancellation_token: predict_image(
            dataset_id=request.dataset_id,
            image_id=request.image_id,
            owner_id=current_user.id,
            cancellation_token=cancellation_token,
        ),
    )

from fastapi import APIRouter, Request

from schemas.predict import PredictRequest, PredictResponse
from services.inference_service import predict_image
from services.request_cancellation import run_cancellable_request

router = APIRouter()


@router.post("/predict", response_model=PredictResponse)
async def predict(
    request: PredictRequest,
    http_request: Request,
) -> PredictResponse:
    return await run_cancellable_request(
        http_request,
        lambda cancellation_token: predict_image(
            dataset_id=request.dataset_id,
            image_id=request.image_id,
            cancellation_token=cancellation_token,
        ),
    )

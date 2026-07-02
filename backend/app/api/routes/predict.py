from fastapi import APIRouter

from schemas.predict import PredictRequest, PredictResponse
from services.inference_service import predict_image

router = APIRouter()


@router.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    return await predict_image(
        dataset_id=request.dataset_id,
        image_id=request.image_id,
    )

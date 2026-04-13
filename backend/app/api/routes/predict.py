from fastapi import APIRouter

from schemas.predict import PredictRequest, PredictResponse
from services.inference_service import predict_image

router = APIRouter()


@router.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    """
    Endpoint para realizar la predicción de una imagen.
    """
    return predict_image(request.image)
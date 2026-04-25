from fastapi import HTTPException

from schemas.predict import PredictResponse
from services.model_service import get_active_model_instance
from utils.image_utils import decode_base64_image


def predict_image(image_b64: str) -> PredictResponse:
    """
    Realiza una predicción sobre una imagen usando el modelo activo.

    La lógica concreta de preprocesado e inferencia queda delegada
    al modelo seleccionado.
    """
    try:
        image = decode_base64_image(image_b64)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        model = get_active_model_instance()
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return model.predict(image)
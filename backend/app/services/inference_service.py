from fastapi import HTTPException

from ml.preprocessing.image_preprocessor import preprocess_image
from schemas.predict import PredictResponse, TopKPrediction
from utils.image_utils import decode_base64_image

def predict_image(image_b64: str) -> PredictResponse:
    """
    Realiza una predicción sobre una imagen.

    Flujo actual:
    1. Decodifica la imagen desde base64
    2. Aplica un preprocesado básico
    3. Devuelve una predicción simulada

    Más adelante aquí se integrará el modelo real con PyTorch.
    """
    try:
        image = decode_base64_image(image_b64)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    processed_image = preprocess_image(image)

    # De momento no usamos `processed_image` para inferencia real,
    # pero el flujo ya queda preparado para ello.
    _ = processed_image
    return PredictResponse(
        prediction="coche",
        confidence=0.9,
        top_k=[
            TopKPrediction(label="coche", probability=0.9),
            TopKPrediction(label="moto", probability=0.1),
        ],
    )
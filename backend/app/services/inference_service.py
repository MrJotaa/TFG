import torch
import torch.nn.functional as F
from fastapi import HTTPException

from ml.preprocessing.image_preprocessor import preprocess_image
from schemas.predict import PredictResponse, TopKPrediction
from services.model_service import (
     get_active_model_instance,
     get_active_model_categories,
)
from utils.image_utils import decode_base64_image

def predict_image(image_b64: str) -> PredictResponse:
    """
    Realiza una predicción real sobre una imagen usando el modelo activo.
    """
    try:
        image = decode_base64_image(image_b64)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    input_tensor = preprocess_image(image)

    try:
        model = get_active_model_instance()
        categories = get_active_model_categories()
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output[0], dim=0)

    top_probs, top_indices = torch.topk(probabilities, k=3)

    top_k_predictions = [
        TopKPrediction(
            label=categories[idx.item()],
            probability=round(prob.item(), 4)
        )
        for prob, idx in zip(top_probs, top_indices)
    ]

    return PredictResponse(
        prediction=top_k_predictions[0].label,
        confidence=top_k_predictions[0].probability,
        top_k=top_k_predictions,
    )
from schemas.predict import PredictResponse, TopKPrediction


def predict_image(image: str) -> PredictResponse:
    """
    Realiza una predicción sobre una imagen.

    Por ahora devuelve un resultado simulado.
    Más adelante aquí se integrará el modelo real.
    """
    return PredictResponse(
        prediction="coche",
        confidence=0.9,
        top_k=[
            TopKPrediction(label="coche", probability=0.9),
            TopKPrediction(label="moto", probability=0.1),
        ],
    )
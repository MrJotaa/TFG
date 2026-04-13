# Aquí definimos la forma de los datos de entrada y salida.
from pydantic import BaseModel

# Representacion individual de cada una de las predicciones del top-k, con su etiqueta y probabilidad asociada.
class TopKPrediction(BaseModel):
    label: str
    probability: float

# Representacion de la solicitud de prediccion, que incluye la imagen codificada en base64.
class PredictRequest(BaseModel):
    image: str

# Representa la respuesta completa del endpoint.
class PredictResponse(BaseModel):
    prediction: str
    confidence: float
    top_k: list[TopKPrediction]
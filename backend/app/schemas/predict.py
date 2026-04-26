# Aquí definimos la forma de los datos de entrada y salida.
from pydantic import BaseModel

# Representacion individual de cada una de las predicciones del top-k, con su etiqueta y probabilidad asociada.
class TopKPrediction(BaseModel):
    label: str
    probability: float

# Representacion de la solicitud de prediccion
# basada en una imagen persistida dentro de un dataset.
class PredictRequest(BaseModel):
    dataset_id: str
    image_id: str

# Representa la respuesta completa del endpoint.
class PredictResponse(BaseModel):
    prediction: str
    confidence: float
    top_k: list[TopKPrediction]
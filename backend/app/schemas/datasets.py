from pydantic import BaseModel

from schemas.classification_metrics import (
    LabelResolutionSummary,
)


class DatasetImage(BaseModel):
    id: str
    filename: str
    url: str | None = None
    label: str | None = None


class DatasetResponse(BaseModel):
    id: str
    name: str
    owner_id: str
    image_count: int
    images: list[DatasetImage]


class DatasetImageLabelUpdate(BaseModel):
    label: str | None = None


class DatasetLabelResolutionResponse(BaseModel):
    """
    Diagnóstico de las etiquetas de un dataset para el modelo
    de clasificación activo.
    """

    dataset_id: str
    model_id: str
    model_name: str
    summary: LabelResolutionSummary

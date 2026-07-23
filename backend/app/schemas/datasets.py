from typing import Literal

from pydantic import BaseModel

from schemas.classification_metrics import (
    LabelResolutionSummary,
)


AnnotationReviewStatus = Literal[
    "unreviewed",
    "reviewed_empty",
    "reviewed_with_objects",
]


class DatasetImage(BaseModel):
    id: str
    filename: str
    url: str | None = None
    label: str | None = None
    width: int | None = None
    height: int | None = None
    annotation_review_status: AnnotationReviewStatus = "unreviewed"


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


class CategoryObjectCount(BaseModel):
    category_id: str
    coco_category_id: int
    name: str
    object_count: int


class DatasetAnnotationCoverage(BaseModel):
    dataset_id: str
    has_annotation_snapshot: bool
    total_images: int
    reviewed_images: int
    unreviewed_images: int
    reviewed_empty_images: int
    reviewed_with_objects_images: int
    coverage_percentage: float
    total_objects: int
    total_categories: int
    objects_by_category: list[CategoryObjectCount]


class CocoAnnotationImportResponse(BaseModel):
    dataset_id: str
    format: Literal["coco"] = "coco"
    replaced_existing_snapshot: bool
    coverage: DatasetAnnotationCoverage

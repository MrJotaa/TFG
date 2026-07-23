from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class DatasetImageIdentity:
    image_id: str
    original_file_name: str
    stored_path: str
    width: int
    height: int


@dataclass(frozen=True)
class CanonicalCategory:
    internal_id: str
    external_id: int
    name: str
    supercategory: str | None


@dataclass(frozen=True)
class CanonicalObjectAnnotation:
    external_id: int
    image_id: str
    category_internal_id: str
    bbox_x1: float
    bbox_y1: float
    bbox_x2: float
    bbox_y2: float
    area: float
    iscrowd: int


@dataclass(frozen=True)
class CanonicalAnnotationSnapshot:
    categories: tuple[CanonicalCategory, ...]
    annotations: tuple[CanonicalObjectAnnotation, ...]
    reviewed_image_ids: frozenset[str]
    image_dimensions: dict[str, tuple[int, int]]


class AnnotationImporter(Protocol):
    format_name: str

    def parse(
        self,
        content: bytes,
        dataset_images: tuple[DatasetImageIdentity, ...],
    ) -> CanonicalAnnotationSnapshot: ...

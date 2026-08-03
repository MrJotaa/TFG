import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real

from bson import ObjectId
from pymongo.errors import PyMongoError

from core.errors import ConflictError, NotFoundError
from db.collections import (
    annotation_categories_collection,
    datasets_collection,
    object_annotations_collection,
)
from services.annotation_importers.base import (
    CanonicalCategory,
    CanonicalObjectAnnotation,
)
from services.dataset_service import (
    DATASET_NOT_FOUND_MESSAGE,
    _validate_dataset_id,
)

DETECTION_SNAPSHOT_MAX_READ_ATTEMPTS = 3
_REVIEW_STATUSES = {
    "unreviewed",
    "reviewed_empty",
    "reviewed_with_objects",
}


class DetectionAnnotationSnapshotError(RuntimeError):
    """Indica que el snapshot persistido no forma un contrato coherente."""


class DetectionAnnotationSnapshotReadError(RuntimeError):
    """Encapsula un fallo externo al materializar el snapshot."""


class DetectionAnnotationSnapshotConflictError(ConflictError):
    """El snapshot activo cambió demasiadas veces durante la lectura."""


@dataclass(frozen=True, slots=True)
class DetectionSnapshotImage:
    image_id: str
    filename: str
    width: int | None
    height: int | None
    review_status: str

    def __post_init__(self) -> None:
        if not isinstance(self.image_id, str) or not self.image_id.strip():
            raise DetectionAnnotationSnapshotError(
                "snapshot image_id must be a non-empty string"
            )
        if not isinstance(self.filename, str) or not self.filename.strip():
            raise DetectionAnnotationSnapshotError(
                f"snapshot image '{self.image_id}' has an invalid filename"
            )
        if self.review_status not in _REVIEW_STATUSES:
            raise DetectionAnnotationSnapshotError(
                f"snapshot image '{self.image_id}' has an invalid review status"
            )
        for field_name, value in (("width", self.width), ("height", self.height)):
            if value is None:
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, Integral)
                or int(value) <= 0
            ):
                raise DetectionAnnotationSnapshotError(
                    f"snapshot image '{self.image_id}' has invalid {field_name}"
                )

        object.__setattr__(self, "image_id", self.image_id.strip())
        object.__setattr__(self, "filename", self.filename.strip())
        if self.width is not None:
            object.__setattr__(self, "width", int(self.width))
        if self.height is not None:
            object.__setattr__(self, "height", int(self.height))


@dataclass(frozen=True, slots=True)
class MaterializedDetectionAnnotationSnapshot:
    dataset_id: str
    dataset_revision: int
    snapshot_id: str | None
    images: tuple[DetectionSnapshotImage, ...]
    categories: tuple[CanonicalCategory, ...]
    annotations: tuple[CanonicalObjectAnnotation, ...]

    @property
    def has_active_snapshot(self) -> bool:
        return self.snapshot_id is not None

    @property
    def reviewed_image_ids(self) -> frozenset[str]:
        return frozenset(
            image.image_id
            for image in self.images
            if image.review_status != "unreviewed"
        )


def _require_mapping(value: object, *, context: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise DetectionAnnotationSnapshotError(f"{context} must be a document")
    return value


def _required_field(
    document: Mapping,
    field_name: str,
    *,
    context: str,
) -> object:
    try:
        return document[field_name]
    except KeyError as error:
        raise DetectionAnnotationSnapshotError(
            f"{context} is missing required field '{field_name}'"
        ) from error


def _document_context(
    document: Mapping,
    *,
    document_type: str,
    index: int,
) -> str:
    identifier = document.get("_id")
    if identifier is None:
        return f"{document_type} document at index {index}"
    return f"{document_type} document '{identifier}'"


def _required_non_empty_string(
    document: Mapping,
    field_name: str,
    *,
    context: str,
) -> str:
    value = _required_field(document, field_name, context=context)
    if not isinstance(value, str) or not value.strip():
        raise DetectionAnnotationSnapshotError(
            f"{context} field '{field_name}' must be a non-empty string"
        )
    return value.strip()


def _required_integer(
    document: Mapping,
    field_name: str,
    *,
    context: str,
) -> int:
    value = _required_field(document, field_name, context=context)
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise DetectionAnnotationSnapshotError(
            f"{context} field '{field_name}' must be an integer"
        )
    return int(value)


def _state(dataset: Mapping) -> tuple[int, str | None]:
    dataset = _require_mapping(dataset, context="dataset snapshot state")
    revision = dataset.get("dataset_revision", 0)
    if isinstance(revision, bool) or not isinstance(revision, Integral):
        raise DetectionAnnotationSnapshotError("dataset_revision must be an integer")
    snapshot_id = dataset.get("active_annotation_snapshot_id")
    if snapshot_id is not None and (
        not isinstance(snapshot_id, str) or not snapshot_id.strip()
    ):
        raise DetectionAnnotationSnapshotError(
            "active_annotation_snapshot_id must be a non-empty string"
        )
    return int(revision), snapshot_id.strip() if snapshot_id is not None else None


def _materialize_images(dataset: Mapping) -> tuple[DetectionSnapshotImage, ...]:
    raw_images = _required_field(dataset, "images", context="dataset document")
    if not isinstance(raw_images, Sequence) or isinstance(
        raw_images, (str, bytes, bytearray)
    ):
        raise DetectionAnnotationSnapshotError(
            "dataset document field 'images' must be a sequence"
        )

    materialized_images = []
    for index, raw_image in enumerate(raw_images):
        context = f"image document at index {index}"
        image = _require_mapping(raw_image, context=context)
        image_id = _required_non_empty_string(
            image,
            "id",
            context=context,
        )
        context = f"image document '{image_id}'"
        materialized_images.append(
            DetectionSnapshotImage(
                image_id=image_id,
                filename=_required_non_empty_string(
                    image,
                    "filename",
                    context=context,
                ),
                width=image.get("width"),
                height=image.get("height"),
                review_status=image.get(
                    "annotation_review_status",
                    "unreviewed",
                ),
            )
        )

    images = tuple(materialized_images)
    image_ids = [image.image_id for image in images]
    if len(image_ids) != len(set(image_ids)):
        raise DetectionAnnotationSnapshotError("dataset contains duplicate image IDs")
    return tuple(sorted(images, key=lambda image: image.image_id))


async def _read_category_documents(
    dataset_id: ObjectId,
    snapshot_id: str,
) -> list[dict]:
    documents = []
    cursor = annotation_categories_collection.find(
        {
            "dataset_id": dataset_id,
            "snapshot_id": snapshot_id,
        }
    ).sort([("coco_category_id", 1), ("_id", 1)])
    async for document in cursor:
        documents.append(document)
    return documents


async def _read_annotation_documents(
    dataset_id: ObjectId,
    snapshot_id: str,
) -> list[dict]:
    documents = []
    cursor = object_annotations_collection.find(
        {
            "dataset_id": dataset_id,
            "snapshot_id": snapshot_id,
        }
    ).sort([("image_id", 1), ("category_id", 1), ("_id", 1)])
    async for document in cursor:
        documents.append(document)
    return documents


def _finite_real(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise DetectionAnnotationSnapshotError(f"{field_name} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise DetectionAnnotationSnapshotError(f"{field_name} must be finite")
    return normalized


def _materialize_category(
    raw_document: object,
    *,
    index: int,
) -> CanonicalCategory:
    initial_context = f"category document at index {index}"
    document = _require_mapping(raw_document, context=initial_context)
    context = _document_context(
        document,
        document_type="category",
        index=index,
    )
    internal_id = _required_field(document, "_id", context=context)
    if internal_id is None or not str(internal_id).strip():
        raise DetectionAnnotationSnapshotError(
            f"{context} field '_id' must be a non-empty identifier"
        )
    persisted_dataset_id = _required_field(
        document,
        "dataset_id",
        context=context,
    )
    if not isinstance(persisted_dataset_id, ObjectId):
        raise DetectionAnnotationSnapshotError(
            f"{context} field 'dataset_id' must be an ObjectId"
        )
    persisted_snapshot_id = _required_field(
        document,
        "snapshot_id",
        context=context,
    )
    if not isinstance(persisted_snapshot_id, str) or not persisted_snapshot_id.strip():
        raise DetectionAnnotationSnapshotError(
            f"{context} field 'snapshot_id' must be a non-empty string"
        )
    supercategory = document.get("supercategory")
    if supercategory is not None and not isinstance(supercategory, str):
        raise DetectionAnnotationSnapshotError(
            f"{context} field 'supercategory' must be a string or null"
        )
    return CanonicalCategory(
        internal_id=str(internal_id),
        external_id=_required_integer(
            document,
            "coco_category_id",
            context=context,
        ),
        name=_required_non_empty_string(
            document,
            "name",
            context=context,
        ),
        supercategory=supercategory,
    )


def _materialize_annotation(
    raw_document: object,
    *,
    index: int,
) -> CanonicalObjectAnnotation:
    initial_context = f"object annotation document at index {index}"
    document = _require_mapping(raw_document, context=initial_context)
    context = _document_context(
        document,
        document_type="object annotation",
        index=index,
    )
    internal_id = _required_field(document, "_id", context=context)
    if internal_id is None or not str(internal_id).strip():
        raise DetectionAnnotationSnapshotError(
            f"{context} field '_id' must be a non-empty identifier"
        )
    persisted_dataset_id = _required_field(
        document,
        "dataset_id",
        context=context,
    )
    if not isinstance(persisted_dataset_id, ObjectId):
        raise DetectionAnnotationSnapshotError(
            f"{context} field 'dataset_id' must be an ObjectId"
        )
    persisted_snapshot_id = _required_field(
        document,
        "snapshot_id",
        context=context,
    )
    if not isinstance(persisted_snapshot_id, str) or not persisted_snapshot_id.strip():
        raise DetectionAnnotationSnapshotError(
            f"{context} field 'snapshot_id' must be a non-empty string"
        )
    category_id = _required_field(document, "category_id", context=context)
    if category_id is None or not str(category_id).strip():
        raise DetectionAnnotationSnapshotError(
            f"{context} field 'category_id' must be a non-empty identifier"
        )
    iscrowd = _required_integer(document, "iscrowd", context=context)
    if iscrowd not in (0, 1):
        raise DetectionAnnotationSnapshotError(
            f"{context} field 'iscrowd' must be 0 or 1"
        )

    def required_finite(field_name: str) -> float:
        value = _required_field(document, field_name, context=context)
        try:
            return _finite_real(value, field_name=field_name)
        except DetectionAnnotationSnapshotError as error:
            raise DetectionAnnotationSnapshotError(
                f"{context} has invalid field '{field_name}'"
            ) from error

    return CanonicalObjectAnnotation(
        external_id=_required_integer(
            document,
            "coco_annotation_id",
            context=context,
        ),
        image_id=_required_non_empty_string(
            document,
            "image_id",
            context=context,
        ),
        category_internal_id=str(category_id),
        bbox_x1=required_finite("bbox_x1"),
        bbox_y1=required_finite("bbox_y1"),
        bbox_x2=required_finite("bbox_x2"),
        bbox_y2=required_finite("bbox_y2"),
        area=required_finite("area"),
        iscrowd=iscrowd,
        object_internal_id=str(internal_id),
    )


def _build_snapshot(
    *,
    dataset_id: ObjectId,
    dataset: dict,
    category_documents: list[dict],
    annotation_documents: list[dict],
) -> MaterializedDetectionAnnotationSnapshot:
    dataset = _require_mapping(dataset, context="dataset document")
    if _required_field(dataset, "_id", context="dataset document") != dataset_id:
        raise DetectionAnnotationSnapshotError(
            "dataset document has an unexpected identifier"
        )
    revision, snapshot_id = _state(dataset)
    images = _materialize_images(dataset)

    categories = tuple(
        _materialize_category(
            document,
            index=index,
        )
        for index, document in enumerate(category_documents)
    )
    category_ids = {category.internal_id for category in categories}
    if len(category_ids) != len(categories):
        raise DetectionAnnotationSnapshotError(
            "annotation snapshot contains duplicate category IDs"
        )

    annotations = tuple(
        _materialize_annotation(
            document,
            index=index,
        )
        for index, document in enumerate(annotation_documents)
    )

    image_by_id = {image.image_id: image for image in images}
    annotation_count_by_image: dict[str, int] = {}
    for annotation in annotations:
        image = image_by_id.get(annotation.image_id)
        if image is None:
            raise DetectionAnnotationSnapshotError(
                f"annotation {annotation.external_id} references an unknown image"
            )
        if annotation.category_internal_id not in category_ids:
            raise DetectionAnnotationSnapshotError(
                f"annotation {annotation.external_id} references an unknown category"
            )
        if image.review_status == "unreviewed":
            raise DetectionAnnotationSnapshotError(
                f"unreviewed image '{image.image_id}' contains annotations"
            )
        if image.width is None or image.height is None:
            raise DetectionAnnotationSnapshotError(
                f"reviewed image '{image.image_id}' has no authoritative dimensions"
            )
        if (
            annotation.bbox_x1 < 0
            or annotation.bbox_y1 < 0
            or annotation.bbox_x1 >= annotation.bbox_x2
            or annotation.bbox_y1 >= annotation.bbox_y2
            or annotation.bbox_x2 > image.width
            or annotation.bbox_y2 > image.height
        ):
            raise DetectionAnnotationSnapshotError(
                f"annotation {annotation.external_id} has incompatible geometry"
            )
        if annotation.area <= 0:
            raise DetectionAnnotationSnapshotError(
                f"annotation {annotation.external_id} has non-positive area"
            )
        if annotation.iscrowd not in (0, 1):
            raise DetectionAnnotationSnapshotError(
                f"annotation {annotation.external_id} has invalid iscrowd"
            )
        annotation_count_by_image[image.image_id] = (
            annotation_count_by_image.get(image.image_id, 0) + 1
        )

    if snapshot_id is not None:
        for image in images:
            if image.review_status != "unreviewed" and (
                image.width is None or image.height is None
            ):
                raise DetectionAnnotationSnapshotError(
                    f"reviewed image '{image.image_id}' has no authoritative dimensions"
                )
            object_count = annotation_count_by_image.get(image.image_id, 0)
            if image.review_status == "reviewed_empty" and object_count:
                raise DetectionAnnotationSnapshotError(
                    f"reviewed-empty image '{image.image_id}' contains annotations"
                )
            if image.review_status == "reviewed_with_objects" and not object_count:
                raise DetectionAnnotationSnapshotError(
                    f"reviewed image '{image.image_id}' has no annotations"
                )

    return MaterializedDetectionAnnotationSnapshot(
        dataset_id=str(dataset_id),
        dataset_revision=revision,
        snapshot_id=snapshot_id,
        images=images,
        categories=tuple(
            sorted(
                categories,
                key=lambda category: (
                    category.external_id,
                    category.internal_id,
                ),
            )
        ),
        annotations=tuple(
            sorted(
                annotations,
                key=lambda annotation: (
                    annotation.image_id,
                    annotation.category_internal_id,
                    annotation.object_internal_id,
                    annotation.external_id,
                ),
            )
        ),
    )


async def read_detection_annotation_snapshot(
    dataset_id: str,
    *,
    max_attempts: int = DETECTION_SNAPSHOT_MAX_READ_ATTEMPTS,
) -> MaterializedDetectionAnnotationSnapshot:
    """Materializa categorías, objetos e imágenes desde un único estado activo."""
    object_id = _validate_dataset_id(dataset_id)
    if isinstance(max_attempts, bool) or not isinstance(max_attempts, Integral):
        raise ValueError("max_attempts must be an integer")
    if int(max_attempts) <= 0:
        raise ValueError("max_attempts must be positive")

    dataset = None
    try:
        for _attempt in range(int(max_attempts)):
            if dataset is None:
                dataset = await datasets_collection.find_one({"_id": object_id})
            if dataset is None:
                raise NotFoundError(DATASET_NOT_FOUND_MESSAGE)

            observed_state = _state(dataset)
            snapshot_id = observed_state[1]
            category_documents = []
            annotation_documents = []
            if snapshot_id is not None:
                category_documents = await _read_category_documents(
                    object_id,
                    snapshot_id,
                )
                annotation_documents = await _read_annotation_documents(
                    object_id,
                    snapshot_id,
                )

            latest_dataset = await datasets_collection.find_one({"_id": object_id})
            if latest_dataset is None:
                raise NotFoundError(DATASET_NOT_FOUND_MESSAGE)
            if _state(latest_dataset) == observed_state:
                return _build_snapshot(
                    dataset_id=object_id,
                    dataset=dataset,
                    category_documents=category_documents,
                    annotation_documents=annotation_documents,
                )
            dataset = latest_dataset
    except (NotFoundError, DetectionAnnotationSnapshotError):
        raise
    except PyMongoError as error:
        raise DetectionAnnotationSnapshotReadError(
            f"could not read annotation snapshot for dataset '{dataset_id}'"
        ) from error

    raise DetectionAnnotationSnapshotConflictError(
        {
            "code": "detection_annotation_snapshot_conflict",
            "message": (
                "Dataset annotations changed repeatedly during evaluation; retry"
            ),
        }
    )

import logging
import os
import uuid
from datetime import datetime, timezone

from bson import ObjectId
from fastapi import UploadFile
from PIL import Image
from pymongo.errors import PyMongoError

from core.errors import (
    BadRequestError,
    ConflictError,
    ForbiddenError,
    NotFoundError,
)
from db.collections import (
    annotation_categories_collection,
    datasets_collection,
    object_annotations_collection,
)
from schemas.datasets import (
    CategoryObjectCount,
    CocoAnnotationImportResponse,
    DatasetAnnotationCoverage,
)
from schemas.users import UserResponse
from services.annotation_importers import get_annotation_importer
from services.annotation_importers.base import DatasetImageIdentity
from services.annotation_importers.coco import CocoImportValidationError
from services.dataset_service import (
    DATASET_NOT_FOUND_MESSAGE,
    _validate_dataset_id,
)

logger = logging.getLogger(__name__)

COCO_MAX_UPLOAD_BYTES = int(os.getenv("COCO_MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
UPLOAD_CHUNK_BYTES = 64 * 1024
ANNOTATION_COVERAGE_MAX_READ_ATTEMPTS = 3


async def _get_owned_dataset_document(
    dataset_id: str,
    current_user: UserResponse,
) -> tuple[ObjectId, dict]:
    object_id = _validate_dataset_id(dataset_id)
    dataset = await datasets_collection.find_one({"_id": object_id})

    if dataset is None:
        raise NotFoundError(DATASET_NOT_FOUND_MESSAGE)
    if str(dataset["owner_id"]) != current_user.id:
        raise ForbiddenError("El conjunto de datos pertenece a otro usuario")

    return object_id, dataset


async def _read_coco_upload(file: UploadFile) -> bytes:
    filename = file.filename or ""
    normalized_filename = filename.replace("\\", "/")

    if (
        not filename
        or "/" in normalized_filename
        or not normalized_filename.lower().endswith(".json")
    ):
        raise CocoImportValidationError(
            "El archivo debe ser un .json con un nombre seguro",
            section="upload",
            field="filename",
            image_name=filename or None,
        )
    content_type = (file.content_type or "").split(";", 1)[0].strip().lower()
    if content_type not in {
        "application/json",
        "application/octet-stream",
        "text/json",
    }:
        raise CocoImportValidationError(
            "El tipo de contenido del archivo COCO no es compatible",
            section="upload",
            field="content_type",
        )

    content = bytearray()
    total_size = 0
    while chunk := await file.read(UPLOAD_CHUNK_BYTES):
        total_size += len(chunk)
        if total_size > COCO_MAX_UPLOAD_BYTES:
            raise CocoImportValidationError(
                (
                    "El JSON COCO supera el límite de subida configurado de "
                    f"{COCO_MAX_UPLOAD_BYTES} bytes"
                ),
                section="upload",
                field="size",
            )
        content.extend(chunk)

    if total_size == 0:
        raise CocoImportValidationError(
            "El archivo JSON COCO está vacío",
            section="upload",
            field="size",
        )

    return bytes(content)


def _authoritative_dimensions(image_metadata: dict) -> tuple[int, int]:
    width = image_metadata.get("width")
    height = image_metadata.get("height")

    if (
        isinstance(width, int)
        and not isinstance(width, bool)
        and width > 0
        and isinstance(height, int)
        and not isinstance(height, bool)
        and height > 0
    ):
        return width, height

    image_path = image_metadata.get("path")
    try:
        with Image.open(image_path) as image:
            width, height = image.size
    except (OSError, TypeError, ValueError) as error:
        raise CocoImportValidationError(
            "No se pudo leer la imagen de referencia del conjunto de datos",
            section="dataset_images",
            field="path",
            image_name=image_metadata.get("filename"),
        ) from error

    if width <= 0 or height <= 0:
        raise CocoImportValidationError(
            "Las dimensiones de la imagen de referencia del conjunto de datos no son válidas",
            section="dataset_images",
            field="dimensions",
            image_name=image_metadata.get("filename"),
        )
    return width, height


def _dataset_image_identities(
    dataset: dict,
) -> tuple[DatasetImageIdentity, ...]:
    identities = []

    for image_metadata in dataset.get("images", []):
        width, height = _authoritative_dimensions(image_metadata)
        identities.append(
            DatasetImageIdentity(
                image_id=image_metadata["id"],
                original_file_name=image_metadata.get(
                    "original_file_name",
                    image_metadata["filename"],
                ),
                stored_path=image_metadata["path"],
                width=width,
                height=height,
            )
        )

    return tuple(identities)


async def _cleanup_snapshot(
    dataset_id: ObjectId,
    snapshot_id: str,
) -> None:
    await annotation_categories_collection.delete_many(
        {
            "dataset_id": dataset_id,
            "snapshot_id": snapshot_id,
        }
    )
    await object_annotations_collection.delete_many(
        {
            "dataset_id": dataset_id,
            "snapshot_id": snapshot_id,
        }
    )


def _build_annotation_coverage(
    dataset_id: str | ObjectId,
    images: list[dict],
    categories: list[dict],
    counts_by_category: dict[str, int],
    *,
    has_annotation_snapshot: bool,
) -> DatasetAnnotationCoverage:
    status_counts = {
        "unreviewed": 0,
        "reviewed_empty": 0,
        "reviewed_with_objects": 0,
    }
    for image in images:
        status = image.get("annotation_review_status", "unreviewed")
        if status not in status_counts:
            status = "unreviewed"
        status_counts[status] += 1

    reviewed_images = (
        status_counts["reviewed_empty"] + status_counts["reviewed_with_objects"]
    )
    ordered_categories = sorted(
        categories,
        key=lambda category: category["coco_category_id"],
    )
    distribution = [
        CategoryObjectCount(
            category_id=str(category["_id"]),
            coco_category_id=category["coco_category_id"],
            name=category["name"],
            object_count=counts_by_category.get(str(category["_id"]), 0),
        )
        for category in ordered_categories
    ]
    total_images = len(images)

    return DatasetAnnotationCoverage(
        dataset_id=str(dataset_id),
        has_annotation_snapshot=has_annotation_snapshot,
        total_images=total_images,
        reviewed_images=reviewed_images,
        unreviewed_images=status_counts["unreviewed"],
        reviewed_empty_images=status_counts["reviewed_empty"],
        reviewed_with_objects_images=status_counts["reviewed_with_objects"],
        coverage_percentage=(
            round(reviewed_images / total_images * 100, 2) if total_images else 0.0
        ),
        total_objects=sum(item.object_count for item in distribution),
        total_categories=len(distribution),
        objects_by_category=distribution,
    )


async def import_coco_annotations(
    dataset_id: str,
    file: UploadFile,
    current_user: UserResponse,
) -> CocoAnnotationImportResponse:
    object_id, dataset = await _get_owned_dataset_document(
        dataset_id,
        current_user,
    )
    if dataset.get("task_type") == "classification":
        raise BadRequestError(
            "Las anotaciones COCO solo están disponibles para conjuntos de datos de detección"
        )
    content = await _read_coco_upload(file)
    importer = get_annotation_importer("coco")
    image_identities = _dataset_image_identities(dataset)

    # Toda la validación ocurre antes de cualquier escritura.
    snapshot = importer.parse(content, image_identities)

    snapshot_id = str(uuid.uuid4())
    category_documents = [
        {
            "_id": category.internal_id,
            "dataset_id": object_id,
            "snapshot_id": snapshot_id,
            "coco_category_id": category.external_id,
            "name": category.name,
            "supercategory": category.supercategory,
        }
        for category in snapshot.categories
    ]
    annotation_documents = [
        {
            "_id": str(uuid.uuid4()),
            "dataset_id": object_id,
            "snapshot_id": snapshot_id,
            "image_id": annotation.image_id,
            "category_id": annotation.category_internal_id,
            "coco_annotation_id": annotation.external_id,
            "bbox_x1": annotation.bbox_x1,
            "bbox_y1": annotation.bbox_y1,
            "bbox_x2": annotation.bbox_x2,
            "bbox_y2": annotation.bbox_y2,
            "area": annotation.area,
            "iscrowd": annotation.iscrowd,
        }
        for annotation in snapshot.annotations
    ]

    annotations_per_image: dict[str, int] = {}
    annotations_per_category: dict[str, int] = {}
    for annotation in snapshot.annotations:
        annotations_per_image[annotation.image_id] = (
            annotations_per_image.get(annotation.image_id, 0) + 1
        )
        annotations_per_category[annotation.category_internal_id] = (
            annotations_per_category.get(annotation.category_internal_id, 0) + 1
        )

    dimensions_by_id = {
        identity.image_id: (identity.width, identity.height)
        for identity in image_identities
    }
    updated_images = []
    for image_metadata in dataset.get("images", []):
        image_id = image_metadata["id"]
        width, height = dimensions_by_id[image_id]
        updated_image = {
            **image_metadata,
            "original_file_name": image_metadata.get(
                "original_file_name",
                image_metadata["filename"],
            ),
            "width": width,
            "height": height,
            "annotation_review_status": "unreviewed",
        }
        if image_id in snapshot.reviewed_image_ids:
            updated_image["annotation_review_status"] = (
                "reviewed_with_objects"
                if annotations_per_image.get(image_id, 0) > 0
                else "reviewed_empty"
            )
        updated_images.append(updated_image)

    import_coverage = _build_annotation_coverage(
        object_id,
        updated_images,
        category_documents,
        annotations_per_category,
        has_annotation_snapshot=True,
    )
    old_snapshot_id = dataset.get("active_annotation_snapshot_id")
    expected_revision = dataset.get("dataset_revision", 0)

    try:
        if category_documents:
            await annotation_categories_collection.insert_many(
                category_documents,
                ordered=False,
            )
        if annotation_documents:
            await object_annotations_collection.insert_many(
                annotation_documents,
                ordered=False,
            )

        activation_filter = {
            "_id": object_id,
            "owner_id": dataset["owner_id"],
            "dataset_revision": expected_revision,
        }
        if old_snapshot_id is None:
            activation_filter["active_annotation_snapshot_id"] = {"$exists": False}
        else:
            activation_filter["active_annotation_snapshot_id"] = old_snapshot_id

        activation_result = await datasets_collection.update_one(
            activation_filter,
            {
                "$set": {
                    "images": updated_images,
                    "active_annotation_snapshot_id": snapshot_id,
                    "annotation_snapshot_format": "coco",
                    "annotation_updated_at": datetime.now(timezone.utc),
                    "task_type": "detection",
                },
                "$inc": {"dataset_revision": 1},
            },
        )
        if activation_result.modified_count != 1:
            raise ConflictError(
                {
                    "code": "annotation_import_conflict",
                    "message": (
                        "El conjunto de datos cambió durante la importación de anotaciones; repite la importación"
                    ),
                    "snapshot_applied": False,
                }
            )
    except (ConflictError, PyMongoError):
        await _cleanup_snapshot(object_id, snapshot_id)
        raise

    # El snapshot sustituido ya es invisible tras el cambio atómico del puntero.
    # Solo ese snapshot puede limpiarse: otros IDs pueden pertenecer a imports
    # concurrentes que todavía estén preparando sus documentos.
    if old_snapshot_id is not None:
        try:
            await _cleanup_snapshot(object_id, old_snapshot_id)
        except PyMongoError:
            logger.exception(
                "Could not remove replaced annotation snapshot %s for dataset %s",
                old_snapshot_id,
                dataset_id,
            )

    return CocoAnnotationImportResponse(
        dataset_id=dataset_id,
        replaced_existing_snapshot=old_snapshot_id is not None,
        coverage=import_coverage,
    )


async def _coverage_for_dataset_document(
    object_id: ObjectId,
    dataset: dict,
) -> DatasetAnnotationCoverage:
    images = dataset.get("images", [])
    snapshot_id = dataset.get("active_annotation_snapshot_id")
    categories = []
    counts_by_category = {}

    if snapshot_id is not None:
        async for category in annotation_categories_collection.find(
            {
                "dataset_id": object_id,
                "snapshot_id": snapshot_id,
            }
        ).sort("coco_category_id", 1):
            categories.append(category)

        annotation_counts = await object_annotations_collection.aggregate(
            [
                {
                    "$match": {
                        "dataset_id": object_id,
                        "snapshot_id": snapshot_id,
                    }
                },
                {
                    "$group": {
                        "_id": "$category_id",
                        "object_count": {"$sum": 1},
                    }
                },
            ]
        )
        async for count in annotation_counts:
            counts_by_category[count["_id"]] = count["object_count"]

    return _build_annotation_coverage(
        object_id,
        images,
        categories,
        counts_by_category,
        has_annotation_snapshot=snapshot_id is not None,
    )


async def get_annotation_coverage(
    dataset_id: str,
    current_user: UserResponse,
) -> DatasetAnnotationCoverage:
    object_id, dataset = await _get_owned_dataset_document(
        dataset_id,
        current_user,
    )

    for _attempt in range(ANNOTATION_COVERAGE_MAX_READ_ATTEMPTS):
        observed_state = (
            dataset.get("dataset_revision", 0),
            dataset.get("active_annotation_snapshot_id"),
        )
        coverage = await _coverage_for_dataset_document(object_id, dataset)
        latest_dataset = await datasets_collection.find_one({"_id": object_id})

        if latest_dataset is None:
            raise NotFoundError(DATASET_NOT_FOUND_MESSAGE)
        if str(latest_dataset["owner_id"]) != current_user.id:
            raise ForbiddenError("El conjunto de datos pertenece a otro usuario")

        latest_state = (
            latest_dataset.get("dataset_revision", 0),
            latest_dataset.get("active_annotation_snapshot_id"),
        )
        if latest_state == observed_state:
            return coverage

        dataset = latest_dataset

    raise ConflictError(
        "El conjunto de datos cambió repetidamente al leer la cobertura de anotaciones; inténtalo de nuevo"
    )

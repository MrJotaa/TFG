import json
import math
import re
import uuid
from pathlib import PurePosixPath

from core.errors import BadRequestError
from services.annotation_importers.base import (
    CanonicalAnnotationSnapshot,
    CanonicalCategory,
    CanonicalObjectAnnotation,
    DatasetImageIdentity,
)


WINDOWS_DRIVE_PATTERN = re.compile(r"^[A-Za-z]:")


class CocoImportValidationError(BadRequestError):
    def __init__(
        self,
        message: str,
        *,
        section: str,
        field: str | None = None,
        coco_id: int | None = None,
        image_name: str | None = None,
    ):
        detail = {
            "code": "invalid_coco_annotations",
            "message": message,
            "section": section,
            "snapshot_applied": False,
        }
        if field is not None:
            detail["field"] = field
        if coco_id is not None:
            detail["coco_id"] = coco_id
        if image_name is not None:
            detail["image_name"] = image_name

        super().__init__(detail)


def normalize_external_file_name(file_name: str) -> str:
    if not isinstance(file_name, str) or not file_name.strip():
        raise ValueError("file_name must be a non-empty string")

    normalized = file_name.strip().replace("\\", "/")

    if any(ord(character) < 32 for character in normalized):
        raise ValueError("file_name contains control characters")
    if normalized.startswith("/") or WINDOWS_DRIVE_PATTERN.match(normalized):
        raise ValueError("absolute paths are not allowed")

    path = PurePosixPath(normalized)

    if any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("unsafe path segments are not allowed")

    return path.as_posix()


def _require_list(root: dict, field: str) -> list:
    value = root.get(field)
    if not isinstance(value, list):
        raise CocoImportValidationError(
            f"'{field}' must be an array",
            section=field,
            field=field,
        )
    return value


def _require_object(value, section: str, index: int) -> dict:
    if not isinstance(value, dict):
        raise CocoImportValidationError(
            f"{section}[{index}] must be an object",
            section=section,
        )
    return value


def _require_integer(
    value,
    *,
    section: str,
    field: str,
    coco_id: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CocoImportValidationError(
            f"'{field}' must be an integer",
            section=section,
            field=field,
            coco_id=coco_id,
        )
    return value


def _require_finite_number(
    value,
    *,
    section: str,
    field: str,
    coco_id: int | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CocoImportValidationError(
            f"'{field}' must be numeric",
            section=section,
            field=field,
            coco_id=coco_id,
        )

    try:
        numeric_value = float(value)
    except OverflowError as error:
        raise CocoImportValidationError(
            f"'{field}' must be finite",
            section=section,
            field=field,
            coco_id=coco_id,
        ) from error
    if not math.isfinite(numeric_value):
        raise CocoImportValidationError(
            f"'{field}' must be finite",
            section=section,
            field=field,
            coco_id=coco_id,
        )
    return numeric_value


class CocoAnnotationImporter:
    format_name = "coco"

    def parse(
        self,
        content: bytes,
        dataset_images: tuple[DatasetImageIdentity, ...],
    ) -> CanonicalAnnotationSnapshot:
        try:
            decoded = content.decode("utf-8")
            root = json.loads(decoded)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise CocoImportValidationError(
                "The uploaded file is not valid UTF-8 COCO JSON",
                section="root",
            ) from error

        if not isinstance(root, dict):
            raise CocoImportValidationError(
                "COCO root must be an object",
                section="root",
            )

        coco_images = _require_list(root, "images")
        coco_annotations = _require_list(root, "annotations")
        coco_categories = _require_list(root, "categories")

        categories, category_by_external_id = self._parse_categories(coco_categories)
        image_by_external_id, reviewed_image_ids, image_dimensions = (
            self._parse_and_associate_images(coco_images, dataset_images)
        )
        annotations = self._parse_annotations(
            coco_annotations,
            image_by_external_id,
            image_dimensions,
            category_by_external_id,
        )

        return CanonicalAnnotationSnapshot(
            categories=tuple(categories),
            annotations=tuple(annotations),
            reviewed_image_ids=frozenset(reviewed_image_ids),
            image_dimensions=image_dimensions,
        )

    def _parse_categories(
        self,
        coco_categories: list,
    ) -> tuple[list[CanonicalCategory], dict[int, CanonicalCategory]]:
        categories = []
        category_by_external_id = {}
        normalized_names = set()

        for index, raw_category in enumerate(coco_categories):
            category = _require_object(raw_category, "categories", index)
            category_id = _require_integer(
                category.get("id"),
                section="categories",
                field="id",
            )

            if category_id in category_by_external_id:
                raise CocoImportValidationError(
                    f"Duplicate category id {category_id}",
                    section="categories",
                    field="id",
                    coco_id=category_id,
                )

            name = category.get("name")
            if not isinstance(name, str) or not name.strip():
                raise CocoImportValidationError(
                    "Category name must be a non-empty string",
                    section="categories",
                    field="name",
                    coco_id=category_id,
                )
            name = name.strip()

            normalized_name = name.casefold()
            if normalized_name in normalized_names:
                raise CocoImportValidationError(
                    f"Duplicate category name '{name}'",
                    section="categories",
                    field="name",
                    coco_id=category_id,
                )

            supercategory = category.get("supercategory")
            if supercategory is not None and not isinstance(supercategory, str):
                raise CocoImportValidationError(
                    "'supercategory' must be a string or null",
                    section="categories",
                    field="supercategory",
                    coco_id=category_id,
                )

            canonical_category = CanonicalCategory(
                internal_id=str(uuid.uuid4()),
                external_id=category_id,
                name=name,
                supercategory=(
                    supercategory.strip() if isinstance(supercategory, str) else None
                ),
            )
            categories.append(canonical_category)
            category_by_external_id[category_id] = canonical_category
            normalized_names.add(normalized_name)

        return categories, category_by_external_id

    def _parse_and_associate_images(
        self,
        coco_images: list,
        dataset_images: tuple[DatasetImageIdentity, ...],
    ) -> tuple[
        dict[int, DatasetImageIdentity],
        set[str],
        dict[str, tuple[int, int]],
    ]:
        exact_matches: dict[str, list[DatasetImageIdentity]] = {}
        basename_matches: dict[str, list[DatasetImageIdentity]] = {}

        for dataset_image in dataset_images:
            try:
                identity = normalize_external_file_name(
                    dataset_image.original_file_name
                )
            except ValueError:
                continue
            exact_matches.setdefault(identity, []).append(dataset_image)
            basename_matches.setdefault(PurePosixPath(identity).name, []).append(
                dataset_image
            )

        image_by_external_id = {}
        mapped_dataset_image_ids = set()
        image_dimensions = {}

        for index, raw_image in enumerate(coco_images):
            image = _require_object(raw_image, "images", index)
            coco_image_id = _require_integer(
                image.get("id"),
                section="images",
                field="id",
            )

            if coco_image_id in image_by_external_id:
                raise CocoImportValidationError(
                    f"Duplicate image id {coco_image_id}",
                    section="images",
                    field="id",
                    coco_id=coco_image_id,
                )

            raw_file_name = image.get("file_name")
            try:
                file_name = normalize_external_file_name(raw_file_name)
            except ValueError as error:
                raise CocoImportValidationError(
                    str(error),
                    section="images",
                    field="file_name",
                    coco_id=coco_image_id,
                    image_name=(
                        raw_file_name if isinstance(raw_file_name, str) else None
                    ),
                ) from error

            candidates = exact_matches.get(file_name, [])
            if not candidates:
                candidates = basename_matches.get(
                    PurePosixPath(file_name).name,
                    [],
                )

            if not candidates:
                raise CocoImportValidationError(
                    "COCO image could not be matched to the dataset",
                    section="images",
                    field="file_name",
                    coco_id=coco_image_id,
                    image_name=file_name,
                )
            if len(candidates) > 1:
                raise CocoImportValidationError(
                    "COCO image match is ambiguous",
                    section="images",
                    field="file_name",
                    coco_id=coco_image_id,
                    image_name=file_name,
                )

            dataset_image = candidates[0]
            if dataset_image.image_id in mapped_dataset_image_ids:
                raise CocoImportValidationError(
                    "The same dataset image is associated more than once",
                    section="images",
                    field="file_name",
                    coco_id=coco_image_id,
                    image_name=file_name,
                )

            declared_width = _require_finite_number(
                image.get("width"),
                section="images",
                field="width",
                coco_id=coco_image_id,
            )
            declared_height = _require_finite_number(
                image.get("height"),
                section="images",
                field="height",
                coco_id=coco_image_id,
            )
            if (
                declared_width <= 0
                or declared_height <= 0
                or not declared_width.is_integer()
                or not declared_height.is_integer()
            ):
                raise CocoImportValidationError(
                    "Image dimensions must be positive integers",
                    section="images",
                    field="width/height",
                    coco_id=coco_image_id,
                    image_name=file_name,
                )
            if (
                int(declared_width) != dataset_image.width
                or int(declared_height) != dataset_image.height
            ):
                raise CocoImportValidationError(
                    (
                        "Declared COCO dimensions do not match the authoritative "
                        "dataset image dimensions"
                    ),
                    section="images",
                    field="width/height",
                    coco_id=coco_image_id,
                    image_name=file_name,
                )

            image_by_external_id[coco_image_id] = dataset_image
            mapped_dataset_image_ids.add(dataset_image.image_id)
            image_dimensions[dataset_image.image_id] = (
                dataset_image.width,
                dataset_image.height,
            )

        return (
            image_by_external_id,
            mapped_dataset_image_ids,
            image_dimensions,
        )

    def _parse_annotations(
        self,
        coco_annotations: list,
        image_by_external_id: dict[int, DatasetImageIdentity],
        image_dimensions: dict[str, tuple[int, int]],
        category_by_external_id: dict[int, CanonicalCategory],
    ) -> list[CanonicalObjectAnnotation]:
        annotations = []
        annotation_ids = set()

        for index, raw_annotation in enumerate(coco_annotations):
            annotation = _require_object(raw_annotation, "annotations", index)
            annotation_id = _require_integer(
                annotation.get("id"),
                section="annotations",
                field="id",
            )
            if annotation_id in annotation_ids:
                raise CocoImportValidationError(
                    f"Duplicate annotation id {annotation_id}",
                    section="annotations",
                    field="id",
                    coco_id=annotation_id,
                )
            annotation_ids.add(annotation_id)

            image_id = _require_integer(
                annotation.get("image_id"),
                section="annotations",
                field="image_id",
                coco_id=annotation_id,
            )
            dataset_image = image_by_external_id.get(image_id)
            if dataset_image is None:
                raise CocoImportValidationError(
                    f"Annotation references unknown COCO image id {image_id}",
                    section="annotations",
                    field="image_id",
                    coco_id=annotation_id,
                )

            category_id = _require_integer(
                annotation.get("category_id"),
                section="annotations",
                field="category_id",
                coco_id=annotation_id,
            )
            category = category_by_external_id.get(category_id)
            if category is None:
                raise CocoImportValidationError(
                    f"Annotation references unknown category id {category_id}",
                    section="annotations",
                    field="category_id",
                    coco_id=annotation_id,
                )

            bbox = annotation.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                raise CocoImportValidationError(
                    "'bbox' must contain [x, y, width, height]",
                    section="annotations",
                    field="bbox",
                    coco_id=annotation_id,
                )
            x, y, width, height = [
                _require_finite_number(
                    value,
                    section="annotations",
                    field=f"bbox[{position}]",
                    coco_id=annotation_id,
                )
                for position, value in enumerate(bbox)
            ]
            if width <= 0 or height <= 0:
                raise CocoImportValidationError(
                    "Bounding-box width and height must be greater than zero",
                    section="annotations",
                    field="bbox",
                    coco_id=annotation_id,
                )

            x2 = x + width
            y2 = y + height
            authoritative_width, authoritative_height = image_dimensions[
                dataset_image.image_id
            ]
            if x < 0 or y < 0 or x2 > authoritative_width or y2 > authoritative_height:
                raise CocoImportValidationError(
                    "Bounding box is outside authoritative image bounds",
                    section="annotations",
                    field="bbox",
                    coco_id=annotation_id,
                    image_name=dataset_image.original_file_name,
                )

            raw_area = annotation.get("area")
            area = (
                width * height
                if raw_area is None
                else _require_finite_number(
                    raw_area,
                    section="annotations",
                    field="area",
                    coco_id=annotation_id,
                )
            )
            if area <= 0:
                raise CocoImportValidationError(
                    "'area' must be greater than zero",
                    section="annotations",
                    field="area",
                    coco_id=annotation_id,
                )

            raw_iscrowd = annotation.get("iscrowd", 0)
            if (
                isinstance(raw_iscrowd, bool)
                or not isinstance(raw_iscrowd, int)
                or raw_iscrowd not in {0, 1}
            ):
                raise CocoImportValidationError(
                    "'iscrowd' must be 0 or 1",
                    section="annotations",
                    field="iscrowd",
                    coco_id=annotation_id,
                )

            annotations.append(
                CanonicalObjectAnnotation(
                    external_id=annotation_id,
                    image_id=dataset_image.image_id,
                    category_internal_id=category.internal_id,
                    bbox_x1=x,
                    bbox_y1=y,
                    bbox_x2=x2,
                    bbox_y2=y2,
                    area=area,
                    iscrowd=raw_iscrowd,
                )
            )

        return annotations

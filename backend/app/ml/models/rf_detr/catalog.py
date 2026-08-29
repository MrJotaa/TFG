from collections.abc import Mapping
from numbers import Integral

from ml.detection_categories import (
    DetectionClass,
    DetectionClassCatalog,
    DetectionClassCatalogError,
)


class RfDetrClassCatalogError(DetectionClassCatalogError):
    """Indica que ``id2label`` no representa un catálogo RF-DETR válido."""


_RF_DETR_PLACEHOLDER_LABELS = frozenset({"n/a"})


def normalize_rf_detr_id2label(
    id2label: Mapping[object, object],
) -> dict[int, str]:
    """Normaliza IDs de configuración sin renumerar el espacio COCO original."""
    if not isinstance(id2label, Mapping):
        raise RfDetrClassCatalogError("RF-DETR id2label must be a mapping")

    normalized: dict[int, str] = {}
    for raw_class_id, raw_name in id2label.items():
        class_id = _normalize_class_id(raw_class_id)
        if class_id in normalized:
            raise RfDetrClassCatalogError(
                f"duplicate RF-DETR class id after normalization: {class_id}"
            )
        normalized[class_id] = _normalize_class_name(raw_name, class_id)

    if not normalized:
        raise RfDetrClassCatalogError("RF-DETR id2label must not be empty")
    return normalized


def _normalize_class_id(raw_class_id: object) -> int:
    if isinstance(raw_class_id, bool):
        raise RfDetrClassCatalogError("RF-DETR class IDs must be integers")
    if isinstance(raw_class_id, Integral):
        class_id = int(raw_class_id)
    elif isinstance(raw_class_id, str):
        try:
            class_id = int(raw_class_id.strip())
        except ValueError as error:
            raise RfDetrClassCatalogError(
                f"invalid RF-DETR class id {raw_class_id!r}"
            ) from error
    else:
        raise RfDetrClassCatalogError(f"invalid RF-DETR class id {raw_class_id!r}")

    if class_id < 0:
        raise RfDetrClassCatalogError("RF-DETR class IDs must not be negative")
    return class_id


def _normalize_class_name(raw_name: object, class_id: int) -> str:
    if not isinstance(raw_name, str) or not raw_name.strip():
        raise RfDetrClassCatalogError(
            f"invalid RF-DETR class name for id {class_id}"
        )
    return raw_name.strip()


def is_rf_detr_placeholder_label(label: str) -> bool:
    return label.strip().casefold() in _RF_DETR_PLACEHOLDER_LABELS


def build_rf_detr_class_catalog(
    *,
    model_id: str,
    model_name: str,
    id2label: Mapping[object, object],
) -> DetectionClassCatalog:
    """
    Adapta ``id2label`` conservando los IDs COCO reales del checkpoint.

    RF-DETR representa los huecos históricos de COCO como ``N/A``. Esos
    placeholders no son clases semánticas y se excluyen del catálogo público,
    pero no se compactan ni se renumeran los IDs restantes.
    """
    normalized = normalize_rf_detr_id2label(id2label)

    classes = tuple(
        DetectionClass(class_id=class_id, name=class_name)
        for class_id, class_name in normalized.items()
        if not is_rf_detr_placeholder_label(class_name)
    )

    try:
        return DetectionClassCatalog(
            model_id=model_id,
            model_name=model_name,
            classes=classes,
        )
    except DetectionClassCatalogError as error:
        raise RfDetrClassCatalogError(
            f"invalid RF-DETR class catalog: {error}"
        ) from error

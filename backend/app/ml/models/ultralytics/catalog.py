from collections.abc import Mapping, Sequence

from ml.detection_categories import (
    DetectionClass,
    DetectionClassCatalog,
    DetectionClassCatalogError,
)


class UltralyticsClassCatalogError(DetectionClassCatalogError):
    """Indica que ``names`` no representa un catálogo Ultralytics válido."""


def build_ultralytics_class_catalog(
    *,
    model_id: str,
    model_name: str,
    names: Mapping[object, object] | Sequence[object],
) -> DetectionClassCatalog:
    """
    Adapta ``names`` de Ultralytics conservando sus IDs reales.

    Los mappings aportan IDs explícitos. Las secuencias usan su posición porque
    esa es la conversión contractual de Ultralytics para ese formato.
    """
    if isinstance(names, Mapping):
        entries = tuple(names.items())
    elif isinstance(names, Sequence) and not isinstance(names, (str, bytes)):
        entries = tuple(enumerate(names))
    else:
        raise UltralyticsClassCatalogError(
            "Ultralytics names must be a mapping or sequence"
        )

    classes = []
    for raw_class_id, raw_name in entries:
        try:
            classes.append(
                DetectionClass(
                    class_id=raw_class_id,
                    name=raw_name,
                )
            )
        except DetectionClassCatalogError as error:
            raise UltralyticsClassCatalogError(
                f"invalid Ultralytics class {raw_class_id!r}: {error}"
            ) from error

    try:
        return DetectionClassCatalog(
            model_id=model_id,
            model_name=model_name,
            classes=tuple(classes),
        )
    except DetectionClassCatalogError as error:
        raise UltralyticsClassCatalogError(
            f"invalid Ultralytics class catalog: {error}"
        ) from error

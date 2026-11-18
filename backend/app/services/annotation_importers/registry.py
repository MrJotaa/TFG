from core.errors import BadRequestError
from services.annotation_importers.base import AnnotationImporter
from services.annotation_importers.coco import CocoAnnotationImporter

_IMPORTERS: dict[str, AnnotationImporter] = {
    "coco": CocoAnnotationImporter(),
}


def get_annotation_importer(format_name: str) -> AnnotationImporter:
    try:
        return _IMPORTERS[format_name]
    except KeyError as error:
        raise BadRequestError(
            f"Formato de anotaciones no compatible: {format_name}"
        ) from error

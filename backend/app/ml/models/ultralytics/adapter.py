import math
from collections.abc import Mapping, Sequence

import torch

from ml.models.detection import (
    Detection,
    DetectionBox,
    DetectionContractError,
    DetectionResult,
)
from ultralytics.engine.results import Results


class DetectionAdapterError(DetectionContractError):
    """Indica que una salida nativa no puede adaptarse con seguridad."""


def _resolve_class_name(
    names: Mapping[int, str] | Sequence[str] | None,
    class_id: int,
) -> str | None:
    if isinstance(names, Mapping):
        class_name = names.get(class_id)
    elif isinstance(names, Sequence) and not isinstance(names, (str, bytes)):
        class_name = names[class_id] if 0 <= class_id < len(names) else None
    else:
        class_name = None

    if class_name is None:
        return None
    if not isinstance(class_name, str) or not class_name.strip():
        raise DetectionAdapterError(
            f"invalid class name for native class id {class_id}"
        )

    return class_name


class UltralyticsDetectionAdapter:
    """
    Convierte un ``Results`` de Ultralytics al contrato canónico.

    Ultralytics proyecta ``boxes.xyxy`` de vuelta a ``result.orig_shape``.
    Por ello se valida esa forma contra la imagen fuente y se conservan las
    coordenadas devueltas, sin duplicar la geometría de resize/letterboxing.
    """

    def adapt(
        self,
        native_result: Results,
        *,
        source_width: int,
        source_height: int,
    ) -> DetectionResult:
        native_shape = tuple(native_result.orig_shape)
        expected_shape = (source_height, source_width)
        if native_shape != expected_shape:
            raise DetectionAdapterError(
                "native result geometry does not match the source image: "
                f"expected {expected_shape}, got {native_shape}"
            )

        boxes = native_result.boxes
        if boxes is None or boxes.xyxy.shape[0] == 0:
            return DetectionResult(
                source_width=source_width,
                source_height=source_height,
            )

        # Una única transferencia/sincronización y conversión a estructuras
        # Python por resultado, no una conversión por cada caja.
        native_rows = torch.cat(
            (
                boxes.xyxy,
                boxes.conf.reshape(-1, 1),
                boxes.cls.reshape(-1, 1),
            ),
            dim=1,
        )
        rows = native_rows.detach().to(device="cpu").tolist()

        detections = tuple(
            detection
            for row_index, row in enumerate(rows)
            if (
                detection := self._adapt_row(
                    row,
                    row_index=row_index,
                    names=native_result.names,
                    source_width=source_width,
                    source_height=source_height,
                )
            )
            is not None
        )

        return DetectionResult(
            source_width=source_width,
            source_height=source_height,
            detections=detections,
        )

    def _adapt_row(
        self,
        row: list[float],
        *,
        row_index: int,
        names: Mapping[int, str] | Sequence[str] | None,
        source_width: int,
        source_height: int,
    ) -> Detection | None:
        if len(row) != 6:
            raise DetectionAdapterError(
                f"native detection {row_index} must contain six values"
            )

        x_min, y_min, x_max, y_max, confidence, native_class_id = row
        if not all(math.isfinite(value) for value in row):
            raise DetectionAdapterError(
                f"native detection {row_index} contains a non-finite value"
            )
        if x_min > x_max or y_min > y_max:
            raise DetectionAdapterError(
                f"native detection {row_index} has an inverted box"
            )
        if not native_class_id.is_integer():
            raise DetectionAdapterError(
                f"native detection {row_index} has a non-integer class id"
            )

        class_id = int(native_class_id)

        try:
            if class_id < 0:
                raise DetectionContractError("class_id must not be negative")

            class_name = _resolve_class_name(names, class_id)
            if not 0.0 <= confidence <= 1.0:
                raise DetectionContractError("confidence must be between 0 and 1")

            clipped_x_min = max(0.0, min(float(source_width), x_min))
            clipped_y_min = max(0.0, min(float(source_height), y_min))
            clipped_x_max = max(0.0, min(float(source_width), x_max))
            clipped_y_max = max(0.0, min(float(source_height), y_max))

            # Ultralytics puede devolver cajas sin superficie al reproyectar
            # o recortar las detecciones. No representan objetos evaluables.
            if clipped_x_min >= clipped_x_max or clipped_y_min >= clipped_y_max:
                return None

            return Detection(
                class_id=class_id,
                class_name=class_name,
                confidence=confidence,
                box=DetectionBox(
                    x_min=clipped_x_min,
                    y_min=clipped_y_min,
                    x_max=clipped_x_max,
                    y_max=clipped_y_max,
                ),
            )
        except DetectionContractError as error:
            raise DetectionAdapterError(
                f"native detection {row_index} is invalid: {error}"
            ) from error

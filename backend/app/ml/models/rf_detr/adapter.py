import math
from collections.abc import Mapping

import torch

from ml.models.detection import (
    Detection,
    DetectionBox,
    DetectionContractError,
    DetectionResult,
)
from ml.models.rf_detr.catalog import is_rf_detr_placeholder_label


class RfDetrDetectionAdapterError(DetectionContractError):
    """Indica que un postprocesado RF-DETR no cumple el contrato canónico."""


class RfDetrDetectionAdapter:
    """Convierte el postprocesado oficial de Transformers a ``DetectionResult``."""

    def adapt(
        self,
        native_result: Mapping[str, torch.Tensor],
        *,
        source_width: int,
        source_height: int,
        id2label: Mapping[int, str],
    ) -> DetectionResult:
        if not isinstance(native_result, Mapping):
            raise RfDetrDetectionAdapterError(
                "RF-DETR postprocessed result must be a mapping"
            )

        scores = native_result.get("scores")
        labels = native_result.get("labels")
        boxes = native_result.get("boxes")

        if not isinstance(scores, torch.Tensor) or scores.ndim != 1:
            raise RfDetrDetectionAdapterError(
                "RF-DETR scores must be a one-dimensional tensor"
            )
        if not scores.is_floating_point():
            raise RfDetrDetectionAdapterError(
                "RF-DETR scores must use a floating-point dtype"
            )
        if not isinstance(labels, torch.Tensor) or labels.ndim != 1:
            raise RfDetrDetectionAdapterError(
                "RF-DETR labels must be a one-dimensional tensor"
            )
        if labels.is_floating_point() or labels.dtype == torch.bool:
            raise RfDetrDetectionAdapterError(
                "RF-DETR labels must use an integer dtype"
            )
        if (
            not isinstance(boxes, torch.Tensor)
            or boxes.ndim != 2
            or boxes.shape[1] != 4
        ):
            raise RfDetrDetectionAdapterError(
                "RF-DETR boxes must have shape [N, 4]"
            )
        if not boxes.is_floating_point():
            raise RfDetrDetectionAdapterError(
                "RF-DETR boxes must use a floating-point dtype"
            )
        if not (len(scores) == len(labels) == len(boxes)):
            raise RfDetrDetectionAdapterError(
                "RF-DETR scores, labels and boxes must have the same length"
            )
        if scores.device != boxes.device or labels.device != boxes.device:
            raise RfDetrDetectionAdapterError(
                "RF-DETR scores, labels and boxes must share the same device"
            )

        if len(scores) == 0:
            return DetectionResult(
                source_width=source_width,
                source_height=source_height,
            )

        rows = (
            torch.cat(
                (
                    boxes,
                    scores.reshape(-1, 1).to(dtype=boxes.dtype),
                    labels.reshape(-1, 1).to(dtype=boxes.dtype),
                ),
                dim=1,
            )
            .detach()
            .to(device="cpu", dtype=torch.float64)
            .tolist()
        )

        detections = []
        for row_index, row in enumerate(rows):
            detection = self._adapt_row(
                row,
                row_index=row_index,
                source_width=source_width,
                source_height=source_height,
                id2label=id2label,
            )
            if detection is not None:
                detections.append(detection)

        return DetectionResult(
            source_width=source_width,
            source_height=source_height,
            detections=tuple(detections),
        )

    def _adapt_row(
        self,
        row: list[float],
        *,
        row_index: int,
        source_width: int,
        source_height: int,
        id2label: Mapping[int, str],
    ) -> Detection | None:
        if len(row) != 6:
            raise RfDetrDetectionAdapterError(
                f"RF-DETR detection {row_index} must contain six values"
            )
        if not all(math.isfinite(value) for value in row):
            raise RfDetrDetectionAdapterError(
                f"RF-DETR detection {row_index} contains a non-finite value"
            )

        x_min, y_min, x_max, y_max, confidence, native_class_id = row
        if x_min >= x_max or y_min >= y_max:
            raise RfDetrDetectionAdapterError(
                f"RF-DETR detection {row_index} has an inverted or degenerate box"
            )
        if not native_class_id.is_integer():
            raise RfDetrDetectionAdapterError(
                f"RF-DETR detection {row_index} has a non-integer class id"
            )

        class_id = int(native_class_id)
        class_name = id2label.get(class_id)
        if class_name is None:
            raise RfDetrDetectionAdapterError(
                f"RF-DETR detection {row_index} references unknown class id {class_id}"
            )
        if not isinstance(class_name, str) or not class_name.strip():
            raise RfDetrDetectionAdapterError(
                f"RF-DETR detection {row_index} has an invalid class name"
            )
        class_name = class_name.strip()
        if is_rf_detr_placeholder_label(class_name):
            return None

        clipped_x_min = max(0.0, min(float(source_width), x_min))
        clipped_y_min = max(0.0, min(float(source_height), y_min))
        clipped_x_max = max(0.0, min(float(source_width), x_max))
        clipped_y_max = max(0.0, min(float(source_height), y_max))

        # RF-DETR boxes are not clamped by the official postprocessor and may
        # extend slightly beyond the source image. Boxes entirely outside the
        # image have no representable positive-area geometry in our contract.
        if clipped_x_min >= clipped_x_max or clipped_y_min >= clipped_y_max:
            return None

        try:
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
            raise RfDetrDetectionAdapterError(
                f"RF-DETR detection {row_index} is invalid: {error}"
            ) from error

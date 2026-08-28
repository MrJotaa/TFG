import math

import pytest

from ml.models.detection import (
    Detection,
    DetectionBox,
    DetectionContractError,
    DetectionResult,
)
from schemas.detection import to_detection_prediction, to_detection_predictions


def _detection(
    *,
    class_id: int = 1,
    class_name: str | None = "person",
    confidence: float = 0.87654321,
    box: DetectionBox | None = None,
) -> Detection:
    return Detection(
        class_id=class_id,
        class_name=class_name,
        confidence=confidence,
        box=box or DetectionBox(1.123456, 2.234567, 30.345678, 40.456789),
    )


def test_detection_result_accepts_empty_prediction_and_is_immutable():
    result = DetectionResult(source_width=640, source_height=480)

    assert result.detections == ()

    with pytest.raises(AttributeError):
        result.source_width = 320


def test_detection_result_copies_collection_and_preserves_full_precision():
    source_detections = [_detection()]

    result = DetectionResult(
        source_width=100,
        source_height=80,
        detections=source_detections,
    )
    source_detections.clear()

    assert isinstance(result.detections, tuple)
    assert len(result.detections) == 1
    assert result.detections[0].confidence == 0.87654321
    assert result.detections[0].box.x_min == 1.123456
    assert result.detections[0].box.y_max == 40.456789


def test_detection_result_accepts_multiple_categories():
    result = DetectionResult(
        source_width=200,
        source_height=100,
        detections=(
            _detection(class_id=0, class_name="cat"),
            _detection(
                class_id=7,
                class_name="truck",
                confidence=0.25,
                box=DetectionBox(100.1, 10.2, 199.9, 99.8),
            ),
        ),
    )

    assert [(item.class_id, item.class_name) for item in result.detections] == [
        (0, "cat"),
        (7, "truck"),
    ]


@pytest.mark.parametrize("width,height", [(0, 10), (-1, 10), (10, 0), (1.5, 10)])
def test_detection_result_rejects_invalid_source_dimensions(width, height):
    with pytest.raises(DetectionContractError):
        DetectionResult(source_width=width, source_height=height)


@pytest.mark.parametrize(
    "confidence",
    [-0.01, 1.01, math.nan, math.inf, -math.inf],
)
def test_detection_rejects_invalid_confidence(confidence):
    with pytest.raises(DetectionContractError):
        _detection(confidence=confidence)


@pytest.mark.parametrize(
    "box",
    [
        (-1.0, 1.0, 2.0, 3.0),
        (3.0, 1.0, 2.0, 4.0),
        (1.0, 4.0, 2.0, 3.0),
        (1.0, 1.0, 1.0, 2.0),
        (1.0, 1.0, 2.0, 1.0),
        (math.nan, 1.0, 2.0, 3.0),
        (1.0, 1.0, math.inf, 3.0),
    ],
)
def test_detection_box_rejects_invalid_geometry(box):
    with pytest.raises(DetectionContractError):
        DetectionBox(*box)


@pytest.mark.parametrize(
    "box",
    [
        DetectionBox(90.0, 10.0, 110.0, 20.0),
        DetectionBox(110.0, 10.0, 120.0, 20.0),
        DetectionBox(10.0, 40.0, 20.0, 60.0),
    ],
)
def test_detection_result_rejects_boxes_outside_source_bounds(box):
    _prepared_detections = (_detection(box=box),)

    with pytest.raises(DetectionContractError):
        DetectionResult(
            source_width=100,
            source_height=50,
            detections=_prepared_detections,
        )


def test_detection_rejects_invalid_class_identity():
    with pytest.raises(DetectionContractError):
        _detection(class_id=-1)

    with pytest.raises(DetectionContractError):
        _detection(class_name=" ")


def test_http_conversion_preserves_legacy_shape_and_rounding():
    canonical = _detection(class_id=4, class_name=None)

    public = to_detection_prediction(canonical)

    assert public.model_dump() == {
        "label": "4",
        "confidence": 0.8765,
        "bbox": {
            "x1": 1.12,
            "y1": 2.23,
            "x2": 30.35,
            "y2": 40.46,
        },
    }
    assert to_detection_predictions((canonical,)) == [public]

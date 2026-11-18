"""C4: validación de trazabilidad y compatibilidad de artefactos anteriores."""

import pytest
from pydantic import ValidationError

from schemas.detection_attack import DetectionAttackResponse
from schemas.detection_evaluation import DetectionEvaluationImageResult
from schemas.detection_square_trace import SquareDetectionExecutionTrace


def _trace():
    return {
        "status": "budget_exhausted",
        "queries": 5,
        "max_queries": 5,
        "cache_hits": 2,
        "seed": 42,
        "attack_revision": "art-1.20.1-square-detection-linf-fixed-references-v1",
        "reference_count": 2,
        "clean_score": 1.0,
        "best_score": 0.75,
        "objective_reached": True,
    }


def test_square_trace_is_json_roundtrippable_and_validated():
    trace = SquareDetectionExecutionTrace.model_validate(_trace())
    assert SquareDetectionExecutionTrace.model_validate_json(trace.model_dump_json()) == trace
    assert trace.objective_reached is True
    with pytest.raises(ValidationError, match="budget"):
        SquareDetectionExecutionTrace.model_validate({**_trace(), "queries": 6})
    with pytest.raises(ValidationError, match="exhaustion"):
        SquareDetectionExecutionTrace.model_validate({**_trace(), "queries": 4})
    with pytest.raises(ValidationError, match="Reference scores"):
        SquareDetectionExecutionTrace.model_validate({**_trace(), "best_score": None})


def test_evaluation_image_trace_is_optional_for_existing_history():
    old = {
        "image_id": "image-1", "filename": "one.jpg",
        "original_detection_count": 1, "adversarial_detection_count": 1,
        "detections_difference": 0,
        "average_original_confidence": 0.9,
        "average_adversarial_confidence": 0.9,
        "average_confidence_difference": 0.0,
        "attack_success": False,
        "perturbation_linf": 0.0, "perturbation_l2": 0.0,
        "original_detections": [], "adversarial_detections": [],
    }
    legacy = DetectionEvaluationImageResult.model_validate(old)
    assert legacy.execution_trace is None
    assert "execution_trace" not in legacy.model_dump(mode="json")
    with_trace = DetectionEvaluationImageResult.model_validate(
        {**old, "execution_trace": _trace()}
    )
    assert with_trace.execution_trace.queries == 5
    assert DetectionEvaluationImageResult.model_validate_json(
        with_trace.model_dump_json()
    ) == with_trace
    assert "execution_trace" in DetectionAttackResponse.model_fields

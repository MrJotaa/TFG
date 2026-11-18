"""C2: contrato de supervivencia de detecciones para Square de caja negra."""

import math

import pytest

from ml.attacks.detection_square_objective import (
    DEFAULT_REFERENCE_CONFIDENCE,
    SquareDetectionObjective,
)
from ml.models.detection import Detection, DetectionBox, DetectionResult


def _det(label, confidence, box):
    return Detection(label, confidence, DetectionBox(*box))


def _result(*detections, width=40, height=32):
    return DetectionResult(width, height, detections)


def _objective(*detections, **kwargs):
    return SquareDetectionObjective.from_clean(_result(*detections), **kwargs)


def test_defaults_use_legacy_reference_threshold_but_not_low_confidence_candidates():
    assert DEFAULT_REFERENCE_CONFIDENCE == 0.25
    good = _det(1, 0.8, (2, 2, 12, 12))
    ignored = _det(1, 0.24, (20, 2, 30, 12))
    objective = _objective(ignored, good)
    assert objective.references == (good,)
    assert objective.score(_result(good)) == 1.0
    assert objective.score(_result(ignored)) == 0.0
    # Las predicciones por debajo del umbral de anclaje sí sirven para medir
    # la caída de confianza; no se descartan al calcular la puntuación.
    fading = _det(1, 0.08, (2, 2, 12, 12))
    assert objective.score(_result(fading)) == pytest.approx(0.1)


def test_clean_is_normalized_to_one_and_high_confidence_does_not_exceed_one():
    a = _det(1, 0.8, (0, 0, 10, 10))
    b = _det(2, 0.4, (12, 2, 25, 16))
    original = _result(a, b)
    obj = SquareDetectionObjective.from_clean(original)
    assert obj.score(original) == pytest.approx(1.0)
    increased = _result(_det(1, 1.0, (0, 0, 10, 10)), _det(2, 0.9, (12, 2, 25, 16)))
    assert obj.score(increased) == pytest.approx(1.0)
    assert not obj.is_success(original)
    assert obj.is_success(_result(_det(1, 0.2, (0, 0, 10, 10)), b))


def test_empty_predictions_wrong_class_and_relocation_all_lower_survival():
    anchor = _det(5, 0.8, (0, 0, 10, 10))
    obj = _objective(anchor)
    assert obj.score(_result()) == 0.0
    assert obj.score(_result(_det(6, 0.99, (0, 0, 10, 10)))) == 0.0
    assert obj.score(_result(_det(5, 0.99, (20, 0, 30, 10)))) == 0.0
    assert obj.score(_result(_det(5, 0.4, (5, 0, 15, 10)))) == pytest.approx(1 / 6)


def test_false_positives_cannot_improve_or_worsen_original_object_survival():
    anchor = _det(0, 0.9, (2, 2, 12, 12))
    obj = _objective(anchor)
    original = _result(anchor)
    with_extras = _result(
        anchor,
        _det(1, 1.0, (2, 2, 12, 12)),
        _det(0, 0.1, (2, 2, 12, 12)),
        _det(0, 0.99, (20, 2, 30, 12)),
    )
    assert obj.score(original) == obj.score(with_extras) == 1.0


def test_one_prediction_cannot_rescue_two_original_objects():
    a = _det(0, 0.8, (0, 0, 10, 10))
    b = _det(0, 0.8, (0, 0, 10, 10))
    obj = _objective(a, b)
    assert obj.score(_result(a, b)) == 1.0
    assert obj.score(_result(a)) == pytest.approx(0.5)


def test_optimal_bipartite_matching_not_greedy_and_independent_of_order():
    a = _det(0, 0.9, (0, 0, 10, 10))
    b = _det(0, 0.9, (10, 0, 20, 10))
    obj = _objective(a, b)
    wide = _det(0, 0.9, (0, 0, 18, 10))
    identical_a = _det(0, 0.9, (0, 0, 10, 10))
    expected = (1.0 + 8.0 / 20.0) / 2.0
    assert obj.score(_result(wide, identical_a)) == pytest.approx(expected)
    assert obj.score(_result(identical_a, wide)) == pytest.approx(expected)


def test_class_mismatch_cannot_steal_assignment():
    first = _det(1, 0.8, (1, 1, 10, 11))
    second = _det(2, 0.9, (1, 1, 10, 11))
    obj = _objective(first, second)
    result = _result(_det(1, 0.8, (1, 1, 10, 11)))
    assert obj.score(result) == pytest.approx(0.5)


def test_references_are_fixed_and_deterministic_independent_of_clean_order():
    first = _det(1, 0.9, (1, 1, 10, 10))
    second = _det(3, 0.7, (12, 1, 20, 10))
    third = _det(4, 0.5, (22, 1, 30, 10))
    obj = _objective(third, first, second, max_references=2)
    reverse = _objective(second, third, first, max_references=2)
    assert obj.references == reverse.references == (first, second)
    lost = _result(third)
    assert obj.score(lost) == 0.0
    assert obj.references == (first, second)
    assert obj.score(_result(first, second)) == 1.0


@pytest.mark.parametrize("value", [0, -0.1, 1.01, float("nan"), float("inf"), True, "0.25"])
def test_invalid_reference_confidence_is_rejected(value):
    with pytest.raises(ValueError, match="min_confidence"):
        _objective(_det(1, 0.9, (1, 1, 10, 10)), min_confidence=value)


@pytest.mark.parametrize("value", [0, -1, 301, 1.5, True, None])
def test_invalid_reference_cap_is_rejected(value):
    with pytest.raises(ValueError, match="max_references"):
        _objective(_det(1, 0.9, (1, 1, 10, 10)), max_references=value)


def test_empty_and_low_confidence_references_are_skipped():
    assert SquareDetectionObjective.from_clean(_result()) is None
    assert _objective(_det(0, 0.1, (0, 0, 10, 10))) is None


def test_prediction_geometry_must_match_reference_source():
    obj = _objective(_det(0, 0.8, (0, 0, 10, 10)))
    with pytest.raises(ValueError, match="geometries"):
        obj.score(_result(_det(0, 0.8, (0, 0, 10, 10)), width=50))
    with pytest.raises(TypeError, match="canonical"):
        obj.score({"boxes": []})
    with pytest.raises(TypeError, match="canonical"):
        SquareDetectionObjective.from_clean({"detections": []})


@pytest.mark.parametrize("value", [-0.2, 1.0, float("nan"), True, "0.8"])
def test_invalid_success_threshold_is_rejected(value):
    obj = _objective(_det(0, 0.9, (0, 0, 10, 10)))
    with pytest.raises(ValueError, match="max_retention"):
        obj.is_success(_result(), max_retention=value)


def test_objective_remains_finite_with_many_candidates_and_is_independent_of_model():
    anchors = tuple(
        _det(
            index % 3,
            0.8,
            (index % 10 * 3, index // 10 * 3,
             index % 10 * 3 + 3, index // 10 * 3 + 3),
        )
        for index in range(100)
    )
    original = _result(*anchors, width=40, height=40)
    obj = SquareDetectionObjective.from_clean(original)
    assert len(obj.references) == 100
    assert math.isfinite(obj.score(original))
    assert obj.score(original) == pytest.approx(1.0)

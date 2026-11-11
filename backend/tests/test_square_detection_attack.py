"""C3: registro, presupuesto estricto y ejecución Square sin gradientes."""

import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from core.cancellation import CancellationToken, OperationCancelled
from core.task_compatibility import require_attack_model_compatibility
from ml.adapters.art_square_detection import (
    DetectionQueryBudgetExhausted,
    SquareDetectionQuerySession,
)
from ml.attacks.registry import create_attack_runner, get_attack, resolve_attack_parameters
from ml.attacks.square_detection import SquareDetectionAttack
from ml.models.detection import Detection, DetectionBox, DetectionResult
from schemas.attack import AttackParameters, AttackType
from utils.reproducibility import attack_parameters_for_image


class _Detector:
    task_type = "detection"

    def __init__(self, model_id="yolo26n", *, weak=False, on_query=None):
        self.id = model_id
        self.loaded = True
        self.calls = 0
        self.weak = weak
        self.on_query = on_query

    def is_loaded(self):
        return self.loaded

    def detect_for_evaluation(self, image, configuration):
        assert configuration.score_floor == 0.001
        self.calls += 1
        if self.on_query:
            self.on_query()
        if self.weak:
            confidence = 0.10
        else:
            pixels = np.asarray(image, dtype=np.float32)
            difference = np.mean(np.abs(pixels - 128.0)) / 255.0
            confidence = float(np.clip(0.9 - 6.0 * difference, 0.01, 0.9))
        return DetectionResult(
            image.width,
            image.height,
            (Detection(0, confidence, DetectionBox(2, 2, 11, 11)),),
        )


def _image(width=20, height=20):
    return Image.new("RGB", (width, height), (128, 128, 128))


def _parameters(**values):
    return resolve_attack_parameters(get_attack(AttackType.SQUARE_DETECTION), AttackParameters(**values))


def _pixel_linf(left, right):
    return float(np.abs(np.asarray(left, dtype=np.int16) - np.asarray(right, dtype=np.int16)).max()) / 255.0


def test_catalog_and_parameter_resolution_support_both_detectors():
    attack = get_attack(AttackType.SQUARE_DETECTION)
    assert isinstance(attack, SquareDetectionAttack)
    assert attack.id == "square_detection"
    assert attack.knowledge_type == "black_box"
    assert attack.revision.startswith("art-1.20.1-")
    assert attack.supported_model_ids == ("yolo26n", "rf_detr_nano")
    assert attack.supported_task_types == ("detection",)
    assert attack.requires_exclusive_model_execution
    assert attack.catalog_entry()["supported_model_ids"] == ["yolo26n", "rf_detr_nano"]
    assert _parameters().model_dump() == {
        "epsilon": 0.03, "max_queries": 50, "steps": 30, "seed": 0,
    }
    for model_id in attack.supported_model_ids:
        require_attack_model_compatibility(attack=attack, model=_Detector(model_id))
    with pytest.raises(Exception, match="compatible"):
        require_attack_model_compatibility(attack=attack, model=_Detector("another_detector"))


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("epsilon", -0.01), ("epsilon", 0.31),
        ("max_queries", 0), ("max_queries", 251), ("max_queries", 3.1),
        ("steps", 0), ("steps", 101),
        ("seed", -1), ("seed", 2147483648),
    ],
)
def test_attack_rejects_invalid_parameters(key, value):
    with pytest.raises(ValueError):
        _parameters(**{key: value})


def test_reproducible_per_image_seed_does_not_change_other_attacks():
    attack = AttackType.SQUARE_DETECTION
    params = _parameters(seed=42)
    first = attack_parameters_for_image(attack_type=attack, parameters=params, image_id="a")
    again = attack_parameters_for_image(attack_type=attack, parameters=params, image_id="a")
    second = attack_parameters_for_image(attack_type=attack, parameters=params, image_id="b")
    assert first.seed == again.seed
    assert first.seed != second.seed
    assert params.seed == 42
    old = attack_parameters_for_image(
        attack_type=AttackType.DETECTION_LOCALIZATION_PGD, parameters=params, image_id="a"
    )
    assert old is params


def test_quantization_limits_actual_rgb_and_cache_counts_queries():
    detector = _Detector()
    session = SquareDetectionQuerySession(detector, _image(), epsilon=0.03, max_queries=2)
    assert session.initialise()
    assert session.queries == detector.calls == 1
    clean = session.clean.copy()
    session.predict(clean)
    assert session.queries == 1
    assert session.cache_hits == 1
    for candidate in (np.zeros_like(clean), np.ones_like(clean)):
        actual = session.quantize(candidate[0])
        assert actual.dtype == np.uint8
        assert _pixel_linf(Image.fromarray(actual), _image()) <= 0.03 + 1e-8
    changed = np.ones_like(clean)
    session.predict(changed)
    assert session.queries == detector.calls == 2
    with pytest.raises(DetectionQueryBudgetExhausted):
        session.predict(np.zeros_like(clean))
    assert session.queries == detector.calls == 2
    assert session.best_score <= 1.0
    assert _pixel_linf(Image.fromarray(session.best_pixels), _image()) <= 0.03 + 1e-8


def test_query_objective_uses_fixed_references_and_no_gradients():
    detector = _Detector()
    session = SquareDetectionQuerySession(detector, _image(), epsilon=0.07, max_queries=3)
    assert session.initialise()
    references = np.empty(1, dtype=object)
    references[0] = session.reference
    initial = session.loss(session.clean, references)
    assert initial[0] == pytest.approx(1.0)
    assert session.queries == 1  # no nueva inferencia para imagen limpia
    worse = np.ones_like(session.clean)
    score = session.loss(worse, references)
    assert score[0] < initial[0]
    assert session.queries == 2
    assert session.success_criterion(np.array([session.reference], dtype=object), references).shape == (1,)
    wrong = np.empty(1, dtype=object)
    wrong[0] = DetectionResult(20, 20, ())
    with pytest.raises(ValueError, match="fixed"):
        session.loss(session.clean, wrong)


def test_skip_on_weak_detections_and_zero_epsilon_does_not_invoke_art():
    attack = get_attack(AttackType.SQUARE_DETECTION)
    original = _image(24, 20)
    for detector, params, expected_queries in (
        (_Detector(weak=True), _parameters(), 1),
        (_Detector(), _parameters(epsilon=0), 0),
        (_Detector(), _parameters(max_queries=1), 0),
    ):
        # A single query budget does not permit a search. No false success.
        runner = create_attack_runner(attack, detector, params)
        result = runner.apply(detector, original, params)
        assert np.array_equal(np.asarray(result["adversarial_image"]), np.asarray(original))
        assert result["perturbation_linf"] == 0
        assert detector.calls == expected_queries


def test_runner_model_isolation_and_cancellation():
    attack = get_attack(AttackType.SQUARE_DETECTION)
    original = _image()
    params = _parameters(max_queries=3)
    token = CancellationToken()
    detector = _Detector()
    runner = create_attack_runner(attack, detector, params, cancellation_token=token)
    assert runner.requires_exclusive_model_execution
    with pytest.raises(ValueError, match="another detector"):
        runner.apply(_Detector("rf_detr_nano"), original, params)
    token.cancel()
    with pytest.raises(OperationCancelled):
        runner.apply(detector, original, params)
    assert detector.calls == 0


def test_cancellation_after_real_query_propagates():
    token = CancellationToken()
    detector = _Detector(on_query=token.cancel)
    session = SquareDetectionQuerySession(detector, _image(), epsilon=0.03, max_queries=3,
                                          cancellation_token=token)
    with pytest.raises(OperationCancelled):
        session.initialise()
    assert detector.calls == session.queries == 1


def test_invalid_detector_output_and_geometry_not_cached():
    detector = _Detector()
    session = SquareDetectionQuerySession(detector, _image(), epsilon=0.03, max_queries=3)
    with pytest.raises(ValueError, match="geometry"):
        session.predict(np.zeros((1, 3, 10, 10), dtype=np.float32))
    candidate = session.clean.copy()
    candidate[0, 0, 1, 1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        session.predict(candidate)
    assert detector.calls == 0


def test_art_budget_abort_recovers_scored_candidate_and_restores_rng(monkeypatch):
    """ART may call both callbacks per step; count ACTUAL detector inference."""
    from ml.attacks import square_detection as production

    class _FakeArtSquare:
        def __init__(self, *, estimator, loss, eps, **kwargs):
            self.loss = loss
            self.eps = eps

        def generate(self, x, y):
            for _ in range(100):
                changed = x + np.random.uniform(-self.eps, self.eps, x.shape).astype(np.float32)
                self.loss(changed, y)
            raise AssertionError("Budget exhaustion should stop ART before this")

    art = types.ModuleType("art")
    attacks = types.ModuleType("art.attacks")
    evasion = types.ModuleType("art.attacks.evasion")
    evasion.SquareAttack = _FakeArtSquare
    monkeypatch.setitem(sys.modules, "art", art)
    monkeypatch.setitem(sys.modules, "art.attacks", attacks)
    monkeypatch.setitem(sys.modules, "art.attacks.evasion", evasion)
    monkeypatch.setattr(production, "create_square_art_estimator", lambda session: object())
    detector = _Detector()
    attack = get_attack(AttackType.SQUARE_DETECTION)
    params = _parameters(max_queries=5, steps=30, seed=7, epsilon=0.08)
    before = np.random.get_state()
    result = attack.apply(detector, _image(), params)
    after = np.random.get_state()
    assert detector.calls == 5
    assert result["perturbation_linf"] <= params.epsilon + 1e-5
    assert _pixel_linf(result["adversarial_image"], _image()) <= params.epsilon + 1e-6
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]

    calls = detector.calls
    again = attack.apply(detector, _image(), params)
    assert detector.calls - calls == 5
    np.testing.assert_array_equal(np.asarray(result["adversarial_image"]),
                                  np.asarray(again["adversarial_image"]))


def test_art_detector_failure_is_not_swallowed(monkeypatch):
    from ml.attacks import square_detection as production

    class _FailArtSquare:
        def __init__(self, **kwargs):
            pass
        def generate(self, x, y):
            raise RuntimeError("unexpected ART failure")

    evasion = types.ModuleType("art.attacks.evasion")
    evasion.SquareAttack = _FailArtSquare
    monkeypatch.setitem(sys.modules, "art", types.ModuleType("art"))
    monkeypatch.setitem(sys.modules, "art.attacks", types.ModuleType("art.attacks"))
    monkeypatch.setitem(sys.modules, "art.attacks.evasion", evasion)
    monkeypatch.setattr(production, "create_square_art_estimator", lambda session: object())
    detector = _Detector()
    with pytest.raises(RuntimeError, match="unexpected ART failure"):
        get_attack(AttackType.SQUARE_DETECTION).apply(detector, _image(),
                                                     _parameters(max_queries=4))
    assert detector.calls == 1


@pytest.mark.parametrize("model_id", ["yolo26n", "rf_detr_nano"])
def test_art_square_production_with_synthetic_detector(model_id):
    pytest.importorskip("art")
    detector = _Detector(model_id)
    image = _image()
    params = _parameters(max_queries=8, steps=12, seed=13, epsilon=0.07)
    attack = get_attack(AttackType.SQUARE_DETECTION)
    runner = create_attack_runner(attack, detector, params)
    first = runner.apply(detector, image, params)
    assert 1 < detector.calls <= params.max_queries
    assert first["adversarial_image"].size == image.size
    assert first["perturbation_image"].size == image.size
    assert _pixel_linf(first["adversarial_image"], image) <= params.epsilon + 1e-6
    assert first["perturbation_linf"] <= params.epsilon + 1e-4
    assert first["perturbation_l2"] >= 0.0
    previous_calls = detector.calls
    second = runner.apply(detector, image, params)
    assert detector.calls - previous_calls <= params.max_queries
    assert np.array_equal(np.asarray(first["adversarial_image"]), np.asarray(second["adversarial_image"]))


@pytest.mark.real_model
@pytest.mark.parametrize("model_id", ["yolo26n", "rf_detr_nano"])
@pytest.mark.skipif(os.getenv("RUN_SQUARE_C3_REAL_SMOKE") != "1",
                    reason="Optional C3 real-model integration")
def test_square_registered_attack_with_real_detector(model_id):
    path = os.getenv("SQUARE_C3_IMAGE_PATH", os.getenv("SQUARE_C1_IMAGE_PATH", ""))
    if not Path(path).is_file():
        pytest.skip("Supply SQUARE_C3_IMAGE_PATH inside backend container")
    from ml.models.definitions import build_model_registry

    model = build_model_registry().get_model(model_id)
    model.load()
    try:
        with Image.open(path) as source:
            image = source.convert("RGB")
        attack = get_attack(AttackType.SQUARE_DETECTION)
        params = _parameters(max_queries=12, steps=3, seed=2026, epsilon=0.03)
        result = create_attack_runner(attack, model, params).apply(model, image, params)
        assert result["adversarial_image"].size == image.size
        assert _pixel_linf(result["adversarial_image"], image) <= 0.03 + 1e-7
        assert result["perturbation_image"].size == image.size
        assert result["perturbation_linf"] <= 0.0301
    finally:
        model.unload()

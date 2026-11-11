"""C1: spike de SquareAttack 1.20.1 sobre el contrato común de detección.

No registra ataques ni cambia producción. Los detectores se consultan exclusivamente
mediante detect_for_evaluation(); cada inferencia real cuenta para el presupuesto.
La función de supervivencia es un proxy de investigación, NO AP/COCO ni éxito final.
"""

import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ml.models.detection import Detection, DetectionBox, DetectionResult
from ml.models.detection_inference import DetectionEvaluationInferenceConfiguration


class _QueryBudgetExhausted(RuntimeError):
    """Se alcanza el límite ANTES de iniciar una inferencia adicional."""


def _iou(left: DetectionBox, right: DetectionBox) -> float:
    intersection = max(0.0, min(left.x_max, right.x_max) - max(left.x_min, right.x_min)) * max(
        0.0, min(left.y_max, right.y_max) - max(left.y_min, right.y_min)
    )
    left_area = (left.x_max - left.x_min) * (left.y_max - left.y_min)
    right_area = (right.x_max - right.x_min) * (right.y_max - right.y_min)
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def _retention(prediction: DetectionResult, reference: DetectionResult) -> float:
    """Menor puntuación = menor supervivencia de objetos de la imagen limpia.

    Una predicción solo puntúa contra anclas de su clase. Las detecciones
    nuevas de otras clases NO se confunden con supervivencia del ancla.
    Esta versión preliminar admite correspondencias muchos-a-uno; C2 revisará
    los emparejamientos ambiguos antes de registrar el ataque.
    """
    if (prediction.source_width, prediction.source_height) != (
        reference.source_width, reference.source_height
    ):
        raise ValueError("Prediction/reference source geometries differ")
    if not reference.detections:
        raise ValueError("Square detection requires non-empty clean references")
    return sum(
        max(
            (
                candidate.confidence * _iou(anchor.box, candidate.box)
                 for candidate in prediction.detections
                 if candidate.class_id == anchor.class_id),
            default=0.0,
        )
        for anchor in reference.detections
    ) / len(reference.detections)


def _as_batch(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))[None]


class _DetectionQueryProbe:
    """Solo test: consultas contadas, caché de inferencias y mejores candidatos."""

    def __init__(self, detector, clean: np.ndarray, *, max_queries: int, eps: float):
        if clean.shape[0] != 1 or clean.shape[1] != 3:
            raise ValueError("C1 only supports one RGB NCHW sample")
        if max_queries < 1 or not 0 <= eps <= 1:
            raise ValueError("Invalid query budget or epsilon")
        self.detector = detector
        self.clean = clean.copy()
        self.eps = float(eps)
        self.max_queries = max_queries
        self.queries = 0
        self.cache_hits = 0
        self.cache = {}
        self.configuration = DetectionEvaluationInferenceConfiguration(score_floor=0.001)
        self.reference = None
        self.baseline = None
        self.best_score = float("inf")
        self.best_image = None

    def _quantize(self, sample: np.ndarray) -> np.ndarray:
        # Proyectar TAMBIÉN después de cuantizar a 8 bits; presupuesto medido
        # sobre la imagen efectivamente enviada a detect_for_evaluation().
        if sample.shape != self.clean.shape[1:]:
            raise ValueError("Input and clean image geometry differ")
        raw = np.rint(np.clip(sample, 0, 1) * 255.0)
        low = np.ceil(np.clip(self.clean[0] - self.eps, 0, 1) * 255.0 - 1e-5)
        high = np.floor(np.clip(self.clean[0] + self.eps, 0, 1) * 255.0 + 1e-5)
        return np.clip(raw, low, high).astype(np.uint8)

    def predict(self, batch: np.ndarray) -> np.ndarray:
        if batch.ndim != 4 or batch.shape[1:] != self.clean.shape[1:]:
            raise ValueError("Expected NCHW in source geometry")
        predictions = np.empty((len(batch),), dtype=object)
        for index, sample in enumerate(batch):
            quantized = self._quantize(sample)
            key = quantized.tobytes()
            if key in self.cache:
                self.cache_hits += 1
                result = self.cache[key]
            else:
                if self.queries >= self.max_queries:
                    raise _QueryBudgetExhausted("Square detection query limit exhausted")
                image = Image.fromarray(np.transpose(quantized, (1, 2, 0)), "RGB")
                result = self.detector.detect_for_evaluation(image, self.configuration)
                if not isinstance(result, DetectionResult):
                    raise TypeError("Expected canonical DetectionResult")
                if (result.source_width, result.source_height) != (
                    image.width, image.height
                ):
                    raise ValueError("Detector changed source image geometry")
                self.queries += 1
                self.cache[key] = result
            predictions[index] = result
            # El mejor candidato se conserva también si ART solo consulta el
            # criterio de éxito y aborta antes de llamar a loss().
            if self.reference is not None:
                score = _retention(result, self.reference)
                if score < self.best_score:
                    self.best_score = score
                    self.best_image = (quantized.astype(np.float32) / 255.0)[None]
        return predictions

    def initialise(self):
        # Llamada limpia incluida en el presupuesto, sin gradientes.
        reference = self.predict(self.clean)[0]
        if not reference.detections:
            return False
        self.reference = reference
        self.baseline = _retention(reference, reference)
        self.best_score = self.baseline
        self.best_image = self.clean.copy()
        return True

    def loss(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        predictions = self.predict(x)
        values = np.array([
            _retention(prediction, ref)
            for prediction, ref in zip(predictions, y)
        ], dtype=np.float32)
        return values

    def criterion(self, y_pred: np.ndarray, y: np.ndarray) -> np.ndarray:
        # Criterio independiente del indicador legacy attack_success.
        values = np.array([
            _retention(prediction, ref)
            for prediction, ref in zip(y_pred, y)
        ])
        return values <= 0.8 * self.baseline


def _run_art_square(probe: _DetectionQueryProbe, *, max_iter: int = 8):
    """ART real con callbacks personalizados; no requiere gradientes."""
    from art.attacks.evasion import SquareAttack
    from art.estimators.estimator import BaseEstimator, NeuralNetworkMixin

    class _Estimator(NeuralNetworkMixin, BaseEstimator):
        def __init__(self):
            super().__init__(channels_first=True, model=probe.detector, clip_values=(0.0, 1.0))

        @property
        def input_shape(self):
            return probe.clean.shape[1:]

        def predict(self, x, batch_size=128, **kwargs):
            return probe.predict(x)

        def fit(self, x, y, batch_size=128, nb_epochs=20, **kwargs):
            raise AssertionError("A black-box detection attack must never fit a detector")

        def get_activations(self, x, layer, batch_size, framework=False):
            raise AssertionError("A black-box attack must not access internal activations")

    if not probe.initialise():
        return probe.clean.copy(), False

    # Evita y=None: ART intentaría generar etiquetas de clasificación a partir
    # de resultados de detección. Los y son anclas canónicas INMUTABLES.
    references = np.empty((1,), dtype=object)
    references[0] = probe.reference
    attack = SquareAttack(
        estimator=_Estimator(),
        norm=np.inf,
        eps=probe.eps,
        p_init=0.05,
        max_iter=max_iter,
        nb_restarts=1,
        batch_size=1,
        loss=probe.loss,
        adv_criterion=probe.criterion,
        verbose=False,
    )
    try:
        attack.generate(probe.clean.copy(), y=references)
    except _QueryBudgetExhausted:
        pass  # ART no permite devolver el mejor candidato al agotar el límite.
    return probe.best_image, True


class _SyntheticDetector:
    """Score de caja decreciente con modificaciones de píxel en un objetivo."""

    def detect_for_evaluation(self, image, configuration):
        assert configuration.score_floor == 0.001
        rgb = np.asarray(image, dtype=np.float32)
        # La intensidad no depende de clases internas, logits o gradientes.
        change = np.mean(np.abs(rgb - 128.0) / 255.0)
        confidence = float(np.clip(0.95 - 16 * change, 0.01, 0.95))
        return DetectionResult(
            source_width=image.width,
            source_height=image.height,
            detections=(Detection(0, confidence, DetectionBox(2, 2, 10, 10)),),
        )


def _detection(class_id, confidence, box, width=32, height=32):
    return DetectionResult(
        source_width=width,
        source_height=height,
        detections=(Detection(class_id, confidence, DetectionBox(*box)),),
    )


def test_retention_decreases_for_disappearance_relocation_class_and_confidence():
    reference = _detection(0, 0.9, (4, 4, 16, 16))
    assert _retention(reference, reference) == pytest.approx(0.9)
    assert _retention(DetectionResult(32, 32, ()), reference) == 0
    assert _retention(_detection(1, 0.9, (4, 4, 16, 16)), reference) == 0
    assert _retention(_detection(0, 0.9, (20, 20, 28, 28)), reference) == 0
    assert _retention(_detection(0, 0.3, (4, 4, 16, 16)), reference) == pytest.approx(0.3)


def test_retention_does_not_reward_false_positives_or_duplicates():
    original = _detection(0, 0.8, (4, 4, 16, 16))
    extra = Detection(2, 0.99, DetectionBox(19, 19, 22, 22))
    duplicate = Detection(0, 0.5, DetectionBox(4, 4, 16, 16))
    prediction = DetectionResult(32, 32, original.detections + (extra, duplicate))
    assert _retention(prediction, original) == pytest.approx(0.8)


def test_retention_requires_reference_and_same_geometry():
    original = _detection(0, 0.8, (4, 4, 16, 16))
    with pytest.raises(ValueError, match="non-empty"):
        _retention(original, DetectionResult(32, 32, ()))
    with pytest.raises(ValueError, match="geometries"):
        _retention(_detection(0, 0.8, (4, 4, 16, 16), width=40), original)


def test_real_detector_budget_counts_only_unique_inferences():
    clean = _as_batch(Image.new("RGB", (16, 16), color=(128, 128, 128)))
    probe = _DetectionQueryProbe(_SyntheticDetector(), clean, max_queries=2, eps=0.06)
    first = probe.predict(clean)
    second = probe.predict(clean)
    assert first[0] == second[0]
    assert probe.queries == 1
    assert probe.cache_hits == 1
    perturbed = clean.copy()
    perturbed[0, 0, :3, :3] = 0.55
    probe.predict(perturbed)
    assert probe.queries == 2
    with pytest.raises(_QueryBudgetExhausted):
        probe.predict(np.zeros_like(clean))
    assert probe.queries == 2


def test_query_saves_best_even_without_explicit_loss_call():
    clean = _as_batch(Image.new("RGB", (20, 20), color=(128, 128, 128)))
    probe = _DetectionQueryProbe(_SyntheticDetector(), clean, max_queries=3, eps=0.08)
    assert probe.initialise()
    perturbed = clean.copy()
    perturbed[:, 0, :6, :6] += 0.05
    probe.predict(perturbed)
    assert probe.best_score < probe.baseline
    assert probe.best_image is not None
    assert float(np.max(np.abs(probe.best_image - clean))) <= 0.08


def test_query_quantization_preserves_strict_linf_budget():
    clean = _as_batch(Image.new("RGB", (16, 16), color=(128, 128, 128)))
    probe = _DetectionQueryProbe(_SyntheticDetector(), clean, max_queries=3, eps=0.03)
    for candidate in (np.zeros_like(clean), np.ones_like(clean)):
        sent = probe._quantize(candidate[0])
        assert float(np.max(np.abs(sent.astype(float) / 255 - clean[0]))) <= 0.03 + 1e-7
    assert probe.queries == 0


def test_square_art_synthetic_respects_budget_and_recovers_best():
    clean = _as_batch(Image.new("RGB", (20, 20), color=(128, 128, 128)))
    probe = _DetectionQueryProbe(_SyntheticDetector(), clean, max_queries=8, eps=0.08)
    state = np.random.get_state()
    try:
        np.random.seed(17)
        best, usable = _run_art_square(probe, max_iter=20)
    finally:
        np.random.set_state(state)
    assert usable
    assert 1 < probe.queries <= probe.max_queries
    assert best.shape == clean.shape
    assert probe.best_score <= probe.baseline
    assert float(np.max(np.abs(best - clean))) <= probe.eps + 1e-6
    assert float(np.max(np.abs(probe._quantize(best[0]).astype(float) / 255 - clean[0]))) <= probe.eps + 1e-6


def test_square_art_stops_at_one_query_without_unscored_output():
    clean = _as_batch(Image.new("RGB", (20, 20), color=(128, 128, 128)))
    probe = _DetectionQueryProbe(_SyntheticDetector(), clean, max_queries=1, eps=0.03)
    best, usable = _run_art_square(probe, max_iter=5)
    assert usable
    assert probe.queries == 1
    np.testing.assert_array_equal(best, clean)


def test_square_art_skips_images_without_clean_detections():
    class _EmptyDetector:
        def detect_for_evaluation(self, image, configuration):
            return DetectionResult(image.width, image.height, ())

    clean = _as_batch(Image.new("RGB", (20, 20), color=(128, 128, 128)))
    probe = _DetectionQueryProbe(_EmptyDetector(), clean, max_queries=4, eps=0.03)
    result, usable = _run_art_square(probe, max_iter=4)
    assert not usable
    assert probe.queries == 1
    np.testing.assert_array_equal(result, clean)


@pytest.mark.real_model
@pytest.mark.parametrize("model_id", ["yolo26n", "rf_detr_nano"])
@pytest.mark.skipif(
    os.getenv("RUN_SQUARE_C1_REAL_SMOKE") != "1",
    reason="Requires explicit C1 real-detector smoke execution",
)
def test_art_square_with_real_detection_interface(model_id):
    """Imagen aportada por el usuario, sin asumir disponibilidad de fixtures externas."""
    path = os.getenv("SQUARE_C1_IMAGE_PATH")
    if not path or not Path(path).is_file():
        pytest.skip("Provide SQUARE_C1_IMAGE_PATH accessible inside backend container")
    from ml.models.definitions import build_model_registry

    model = build_model_registry().get_model(model_id)
    model.load()
    try:
        with Image.open(path) as source:
            image = source.convert("RGB")
        clean = _as_batch(image)
        probe = _DetectionQueryProbe(model, clean, max_queries=12, eps=0.03)
        state = np.random.get_state()
        try:
            np.random.seed(2026)
            best, usable = _run_art_square(probe, max_iter=3)
        finally:
            np.random.set_state(state)
        assert usable, (
            "Choose a real image with at least one detection for BOTH models; "
            "the ART pathway was not exercised on an empty reference set"
        )
        assert 1 < probe.queries <= 12
        assert best.shape == clean.shape
        assert probe.best_score <= probe.baseline + 1e-7
        assert np.max(np.abs(best - clean)) <= 0.03 + 1e-6

        # Comparación exploratoria con ruido uniforme del mismo L∞ y el mismo
        # número de consultas REALES (imagen limpia incluida). No se exige que
        # Square gane: esta comparación es la puerta experimental de C1.
        baseline_probe = _DetectionQueryProbe(
            model, clean, max_queries=probe.queries, eps=0.03
        )
        assert baseline_probe.initialise()
        rng = np.random.default_rng(2026)
        refs = np.empty((1,), dtype=object)
        refs[0] = baseline_probe.reference
        for _ in range(probe.queries - 1):
            noise = rng.uniform(-0.03, 0.03, size=clean.shape).astype(np.float32)
            candidate = np.clip(clean + noise, 0, 1)
            baseline_probe.loss(candidate, refs)
        assert baseline_probe.queries <= probe.queries
        print(
            f"C1 {model_id}: clean={probe.baseline:.5f}, "
            f"square={probe.best_score:.5f}, "
            f"random={baseline_probe.best_score:.5f}, "
            f"square_queries={probe.queries}, random_queries={baseline_probe.queries}"
        )
    finally:
        model.unload()

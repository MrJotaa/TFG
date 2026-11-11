"""Adaptación de inferencia de detección al contrato score-based de SquareAttack.

El estado de consultas pertenece a UNA imagen: no se almacena en el detector
compartido ni en el ataque registrado. Todas las puntuaciones proceden de
`detect_for_evaluation`, sin acceso a logits ni gradientes.
"""

from hashlib import blake2b
from math import floor, isfinite

import numpy as np
import torch
from PIL import Image

from ml.attacks.detection_square_objective import SquareDetectionObjective
from ml.models.detection import DetectionResult
from ml.models.detection_inference import DetectionEvaluationInferenceConfiguration


class DetectionQueryBudgetExhausted(RuntimeError):
    """La siguiente consulta superaría el presupuesto de inferencias reales."""


class SquareDetectionQuerySession:
    """Contabiliza consultas y conserva el mejor candidato *evaluado*.

    Los candidatos se proyectan al presupuesto L∞ DESPUÉS de cuantizar. Una
    consulta repetida de la misma imagen cuantizada utiliza la caché y no
    consume una inferencia. La imagen limpia cuenta como la primera consulta.
    """

    def __init__(self, detector, image: Image.Image, *, epsilon: float, max_queries: int,
                 cancellation_token=None):
        if not isinstance(image, Image.Image):
            raise TypeError("Square detection input must be a PIL image")
        if not isfinite(epsilon) or not 0 <= epsilon <= 0.3:
            raise ValueError("Invalid Square detection epsilon")
        if isinstance(max_queries, bool) or not isinstance(max_queries, int) or not 1 <= max_queries <= 250:
            raise ValueError("Invalid Square detection query budget")
        self.detector = detector
        self.original = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        self.clean = (self.original.astype(np.float32).transpose(2, 0, 1)[None] / 255.0)
        self.epsilon = float(epsilon)
        self.max_queries = max_queries
        self.cancellation_token = cancellation_token
        self.queries = 0
        self.cache_hits = 0
        self._cache: dict[bytes, DetectionResult] = {}
        self.configuration = DetectionEvaluationInferenceConfiguration(score_floor=0.001)
        self.reference: DetectionResult | None = None
        self.objective: SquareDetectionObjective | None = None
        self.best_score = float("inf")
        self.clean_score: float | None = None
        self.best_pixels = self.original.copy()

    def check_cancelled(self):
        if self.cancellation_token is not None:
            self.cancellation_token.raise_if_cancelled()

    def quantize(self, candidate: np.ndarray) -> np.ndarray:
        """CHW float -> HWC uint8; garantiza el presupuesto en píxeles finales."""
        if candidate.shape != self.clean.shape[1:]:
            raise ValueError("Square candidate and source geometry differ")
        if not np.isfinite(candidate).all():
            raise ValueError("Square candidate must contain finite pixels")
        raw = np.rint(np.clip(candidate, 0.0, 1.0) * 255.0)
        # La imagen limpia es de 8 bits: el máximo delta entero permitido es
        # floor(epsilon*255). No se da margen extra por redondeo de ART.
        levels = floor(self.epsilon * 255.0 + 1e-9)
        low = np.maximum(self.original.astype(np.int16) - levels, 0)
        high = np.minimum(self.original.astype(np.int16) + levels, 255)
        return np.clip(raw.transpose(1, 2, 0), low, high).astype(np.uint8)

    def predict(self, batch: np.ndarray) -> np.ndarray:
        self.check_cancelled()
        if not isinstance(batch, np.ndarray) or batch.ndim != 4 or batch.shape[1:] != self.clean.shape[1:]:
            raise ValueError("Square detector expects NCHW RGB source geometry")
        outputs = np.empty(len(batch), dtype=object)
        for index, candidate in enumerate(batch):
            self.check_cancelled()
            pixels = self.quantize(candidate)
            # No retenemos una copia RGB completa como clave por consulta.
            key = blake2b(pixels.tobytes(), digest_size=32).digest()
            if key in self._cache:
                self.cache_hits += 1
                prediction = self._cache[key]
            else:
                if self.queries >= self.max_queries:
                    raise DetectionQueryBudgetExhausted("Square detector query budget exhausted")
                image = Image.fromarray(pixels, mode="RGB")
                with torch.no_grad():
                    prediction = self.detector.detect_for_evaluation(image, self.configuration)
                # Cuenta incluso si la predicción viola el contrato (ya se ha
                # realizado la inferencia); el fallo se propaga sin ocultarlo.
                self.queries += 1
                self.check_cancelled()
                if not isinstance(prediction, DetectionResult):
                    raise TypeError("Detection model must return DetectionResult")
                if (prediction.source_width, prediction.source_height) != image.size:
                    raise ValueError("Detector changed source geometry")
                self._cache[key] = prediction
            outputs[index] = prediction
            if self.objective is not None:
                score = self.objective.score(prediction)
                if score < self.best_score:
                    self.best_score = score
                    self.best_pixels = pixels.copy()
        return outputs

    def initialise(self) -> bool:
        """Consulta limpia una vez y congela las pseudorreferencias."""
        self.check_cancelled()
        prediction = self.predict(self.clean)[0]
        objective = SquareDetectionObjective.from_clean(prediction)
        if objective is None:
            return False
        self.reference = prediction
        self.objective = objective
        self.best_score = objective.score(prediction)
        self.clean_score = self.best_score
        self.best_pixels = self.original.copy()
        return True

    def _scores(self, predictions: np.ndarray, references: np.ndarray) -> np.ndarray:
        if self.objective is None or self.reference is None:
            raise RuntimeError("Square detection objective is not initialised")
        if len(predictions) != len(references) or any(
            reference != self.reference for reference in references
        ):
            raise ValueError("Square detection references must remain fixed")
        return np.asarray([self.objective.score(pred) for pred in predictions], dtype=np.float32)

    def loss(self, images: np.ndarray, references: np.ndarray) -> np.ndarray:
        return self._scores(self.predict(images), references)

    def success_criterion(self, predictions: np.ndarray, references: np.ndarray) -> np.ndarray:
        # Solo parada operativa de ART, NO la métrica histórica attack_success.
        return self._scores(predictions, references) <= 0.8


def create_square_art_estimator(session: SquareDetectionQuerySession):
    """Estimador de ART 1.20.1 sin ninguna operación diferenciable."""
    from art.estimators.estimator import BaseEstimator, NeuralNetworkMixin

    class _Estimator(NeuralNetworkMixin, BaseEstimator):
        def __init__(self):
            super().__init__(
                channels_first=True,
                model=session.detector,
                clip_values=(0.0, 1.0),
            )

        @property
        def input_shape(self):
            return session.clean.shape[1:]

        def predict(self, x, batch_size=128, **kwargs):
            return session.predict(x)

        def fit(self, x, y, batch_size=128, nb_epochs=20, **kwargs):
            raise RuntimeError("A black-box detection attack must never train a model")

        def get_activations(self, x, layer, batch_size, framework=False):
            raise RuntimeError("A black-box detection attack must not use activations")

    return _Estimator()

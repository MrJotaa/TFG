"""Square-style Detection, caja negra con presupuesto real de consultas.

Usa SquareAttack de ART 1.20.1 y la función de supervivencia de C2. El mejor
candidato sólo se acepta tras pasar por la inferencia cuantizada del detector.
No optimiza la pérdida de clasificación ni utiliza gradientes de ningún modelo.
"""

from dataclasses import dataclass
from threading import Lock

import numpy as np
import torch
from PIL import Image

from ml.adapters.art_square_detection import (
    DetectionQueryBudgetExhausted,
    SquareDetectionQuerySession,
    create_square_art_estimator,
)
from ml.attacks.base import AttackParameterSpec, BaseAdversarialAttack
from ml.attacks.presentation import perturbation_to_visualization

# ART emplea el generador global de NumPy. Protegemos el ciclo seed/generate/
# restore entre ejecuciones de este ataque, sin compartir estado entre imágenes.
_SQUARE_ART_RNG_LOCK = Lock()


@dataclass(frozen=True, slots=True)
class _SquareDetectionRunner:
    attack: "SquareDetectionAttack"
    model_wrapper: object
    cancellation_token: object | None
    requires_exclusive_model_execution = True

    def apply(self, model_wrapper, image, parameters, original_class_index=None):
        if model_wrapper is not self.model_wrapper:
            raise ValueError("Square runner cannot be reused with another detector")
        return self.attack._apply(model_wrapper, image, parameters, self.cancellation_token)


class SquareDetectionAttack(BaseAdversarialAttack):
    id = "square_detection"
    name = "Square Attack (detección)"
    description = (
        "Ataque de caja negra con perturbación L∞ acotada y máximo estricto "
        "de consultas al detector; reduce la supervivencia de detecciones originales"
    )
    knowledge_type = "black_box"
    revision = "art-1.20.1-square-detection-linf-fixed-references-v1"
    supported_task_types = ("detection",)
    supported_model_ids = ("yolo26n", "rf_detr_nano")
    requires_exclusive_model_execution = True
    parameter_specs = (
        AttackParameterSpec(
            key="epsilon", label="Épsilon (ε)", summary_label="ε",
            description="Presupuesto L∞ sobre la imagen RGB de 8 bits",
            value_type="float", default=0.03, control="range",
            minimum=0.0, maximum=0.3, step=0.01,
        ),
        AttackParameterSpec(
            key="max_queries", label="Consultas máximas", summary_label="consultas",
            description="Máximo de inferencias reales por imagen, incluida la original",
            value_type="integer", default=50, control="number",
            minimum=1, maximum=250, step=1,
        ),
        AttackParameterSpec(
            key="steps", label="Iteraciones máximas", summary_label="iter",
            description="Límite de iteraciones internas de ART, distinto de las consultas",
            value_type="integer", default=30, control="number",
            minimum=1, maximum=100, step=1,
        ),
        AttackParameterSpec(
            key="seed", label="Semilla", summary_label="semilla",
            description="Semilla reproducible de la búsqueda aleatoria",
            value_type="integer", default=0, control="number",
            minimum=0, maximum=2_147_483_647, step=1,
        ),
    )

    def create_runner(self, model_wrapper, parameters, cancellation_token=None):
        return _SquareDetectionRunner(self, model_wrapper, cancellation_token)

    def apply(self, model_wrapper, image, parameters, original_class_index=None):
        return self._apply(model_wrapper, image, parameters, None)

    @staticmethod
    def _check_cancelled(token):
        if token is not None:
            token.raise_if_cancelled()

    def _apply(self, model_wrapper, image, parameters, cancellation_token):
        self._check_cancelled(cancellation_token)
        if getattr(model_wrapper, "id", None) not in self.supported_model_ids:
            raise ValueError("Square Attack sólo admite YOLO26n y RF-DETR Nano")
        if not model_wrapper.is_loaded():
            raise ValueError("Model is not loaded")
        if not isinstance(image, Image.Image):
            raise TypeError("Square detection input must be a PIL image")

        epsilon = float(parameters.epsilon)
        max_queries = int(parameters.max_queries)
        steps = int(parameters.steps)
        seed = int(parameters.seed)
        if not (0.0 <= epsilon <= 0.3 and 1 <= max_queries <= 250 and
                1 <= steps <= 100 and 0 <= seed <= 2_147_483_647):
            raise ValueError("Invalid Square detection parameters")

        session = SquareDetectionQuerySession(
            model_wrapper, image, epsilon=epsilon, max_queries=max_queries,
            cancellation_token=cancellation_token,
        )
        if epsilon > 0 and max_queries > 1 and session.initialise():
            self._check_cancelled(cancellation_token)
            from art.attacks.evasion import SquareAttack

            references = np.empty(1, dtype=object)
            references[0] = session.reference
            estimator = create_square_art_estimator(session)
            art_attack = SquareAttack(
                estimator=estimator,
                norm=np.inf,
                eps=epsilon,
                p_init=0.05,
                max_iter=steps,
                nb_restarts=1,
                batch_size=1,
                loss=session.loss,
                adv_criterion=session.success_criterion,
                verbose=False,
            )
            # Se captura EXCLUSIVAMENTE la señal de presupuesto agotado. Los
            # errores de detector/ART/cancelación deben llegar al servicio.
            with _SQUARE_ART_RNG_LOCK:
                state = np.random.get_state()
                try:
                    np.random.seed(seed)
                    try:
                        art_attack.generate(session.clean.copy(), y=references)
                    except DetectionQueryBudgetExhausted:
                        pass
                finally:
                    np.random.set_state(state)

        self._check_cancelled(cancellation_token)
        # session.best_pixels procede siempre de una inferencia REAL, salvo
        # epsilon=0 o falta de referencias, donde se devuelve la imagen fuente.
        pixels = session.best_pixels
        adversarial_image = Image.fromarray(pixels.copy(), mode="RGB")
        original = torch.from_numpy(session.original.copy()).permute(2, 0, 1).float() / 255.0
        final = torch.from_numpy(pixels.copy()).permute(2, 0, 1).float() / 255.0
        delta = (final - original).unsqueeze(0)
        linf, l2 = self._perturbation_norms(delta)
        if linf > epsilon + 1e-6:
            raise RuntimeError("Square result violates its 8-bit L∞ budget")
        return {
            "adversarial_image": adversarial_image,
            "perturbation_image": perturbation_to_visualization(delta),
            "perturbation_linf": round(linf, 4),
            "perturbation_l2": round(l2, 4),
        }

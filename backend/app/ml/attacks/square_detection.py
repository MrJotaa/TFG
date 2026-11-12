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
from schemas.detection_square_trace import SquareDetectionExecutionTrace

# ART emplea el generador global de NumPy. Protegemos el ciclo seed/generate/
# restore entre ejecuciones de este ataque, sin compartir estado entre imágenes.
_SQUARE_ART_RNG_LOCK = Lock()


def _square_initial_fraction(width: int, height: int) -> float:
    """Limit ART's initial square to fit both sides of a source image.

    ART 1.20.1 samples square origins with randint(0, side - tile),
    so the rounded tile must be strictly smaller than the shortest side.
    The schedule only decreases p_init in subsequent iterations.
    """
    shortest = min(width, height)
    if shortest < 2:
        raise ValueError("Square Attack requires both image dimensions >= 2")
    return min(0.05, (shortest - 1) ** 2 / (width * height))


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
    revision = "art-1.20.1-square-detection-linf-fixed-references-v2"
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
            description=("Consultas internas máximas de Square, incluida la referencia "
                         "limpia; no incluye métricas o visualización posterior"),
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
        if epsilon == 0:
            status = "skipped_zero_epsilon"
        elif max_queries == 1:
            status = "skipped_insufficient_budget"
        elif min(image.size) < 2:
            status = "skipped_unsupported_geometry"
        elif not session.initialise():
            status = "skipped_no_references"
        else:
            status = "completed"
            self._check_cancelled(cancellation_token)
            from art.attacks.evasion import SquareAttack

            references = np.empty(1, dtype=object)
            references[0] = session.reference
            estimator = create_square_art_estimator(session)
            art_attack = SquareAttack(
                estimator=estimator,
                norm=np.inf,
                eps=epsilon,
                p_init=_square_initial_fraction(*image.size),
                max_iter=steps,
                nb_restarts=1,
                batch_size=1,
                loss=session.loss,
                adv_criterion=session.success_criterion,
                verbose=False,
            )
            # ART usa el RNG global de NumPy. Mientras otro runner tenga el
            # bloqueo, comprobar periódicamente la cancelación; no esperar a
            # que termine la búsqueda ajena para atenderla.
            while not _SQUARE_ART_RNG_LOCK.acquire(timeout=0.05):
                self._check_cancelled(cancellation_token)
            try:
                self._check_cancelled(cancellation_token)
                state = np.random.get_state()
                try:
                    np.random.seed(seed)
                    try:
                        art_attack.generate(session.clean.copy(), y=references)
                    except DetectionQueryBudgetExhausted:
                        status = "budget_exhausted"
                finally:
                    np.random.set_state(state)
            finally:
                _SQUARE_ART_RNG_LOCK.release()

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
        # Una traza por imagen, sin estado compartido. No incluye inferencias
        # adicionales de visualización o métricas posteriores al ataque.
        score = session.best_score if session.objective is not None else None
        trace = SquareDetectionExecutionTrace(
            status=status,
            queries=session.queries,
            max_queries=max_queries,
            cache_hits=session.cache_hits,
            seed=seed,
            attack_revision=self.revision,
            reference_count=(len(session.objective.references) if session.objective else 0),
            clean_score=(session.clean_score if session.objective else None),
            best_score=score,
            objective_reached=(score is not None and score <= 0.8),
        )
        return {
            "adversarial_image": adversarial_image,
            "perturbation_image": perturbation_to_visualization(delta),
            "perturbation_linf": round(linf, 4),
            "perturbation_l2": round(l2, 4),
            "execution_trace": trace.model_dump(mode="json"),
        }

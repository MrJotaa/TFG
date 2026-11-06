"""PGD L-infinito contra la localización de RF-DETR.

Las referencias son detecciones de la imagen SIN perturbar y permanecen fijas
entre iteraciones. El objetivo diferenciable L1 + GIoU vive en el detector
(A3); este módulo sólo implementa el ascenso proyectado y los artefactos.
"""

from dataclasses import dataclass
from math import floor

import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image

from ml.attacks.base import AttackParameterSpec, BaseAdversarialAttack
from ml.attacks.presentation import perturbation_to_visualization
from ml.models.detection_localization import DetectionLocalizationReferences

# Umbral de las predicciones usadas como pseudorreferencia; coincide con el
# umbral legacy de detección. No se reestima durante las iteraciones PGD.
_REFERENCE_CONFIDENCE = 0.25


@dataclass(frozen=True, slots=True)
class _LocalizationPGDRunner:
    """Captura la cancelación por ejecución, nunca en el ataque registrado."""

    attack: "DetectionLocalizationPGDAttack"
    cancellation_token: object | None
    requires_exclusive_model_execution = True

    def apply(self, model_wrapper, image, parameters, original_class_index=None):
        return self.attack._apply(
            model_wrapper=model_wrapper,
            image=image,
            parameters=parameters,
            cancellation_token=self.cancellation_token,
        )


class DetectionLocalizationPGDAttack(BaseAdversarialAttack):
    id = "detection_localization_pgd"
    name = "PGD de localización"
    description = (
        "Ascenso PGD sobre las pérdidas L1 y GIoU de RF-DETR usando "
        "cajas fijas detectadas en la imagen original"
    )
    knowledge_type = "white_box"
    revision = "rf-detr-localization-pgd-linf-pseudo-v1"
    supported_task_types = ("detection",)
    supported_model_ids = ("rf_detr_nano",)
    requires_exclusive_model_execution = True
    parameter_specs = (
        AttackParameterSpec(
            key="epsilon",
            label="Épsilon (ε)",
            summary_label="ε",
            description="Presupuesto máximo de perturbación L∞",
            value_type="float",
            default=0.03,
            control="range",
            minimum=0.0,
            maximum=0.3,
            step=0.01,
        ),
        AttackParameterSpec(
            key="alpha",
            label="Alfa (α)",
            summary_label="α",
            description="Tamaño de paso de ascenso por iteración",
            value_type="float",
            default=0.01,
            control="range",
            minimum=0.0,
            maximum=0.1,
            step=0.001,
        ),
        AttackParameterSpec(
            key="steps",
            label="Iteraciones",
            summary_label="iter",
            description="Número de pasos de ascenso proyectado",
            value_type="integer",
            default=10,
            control="number",
            minimum=1,
            maximum=100,
            step=1,
        ),
    )

    def create_runner(self, model_wrapper, parameters, cancellation_token=None):
        return _LocalizationPGDRunner(self, cancellation_token)

    def apply(self, model_wrapper, image, parameters, original_class_index=None):
        return self._apply(
            model_wrapper=model_wrapper,
            image=image,
            parameters=parameters,
            cancellation_token=None,
        )

    @staticmethod
    def _check_cancelled(token):
        if token is not None:
            token.raise_if_cancelled()

    def _apply(self, *, model_wrapper, image: Image.Image, parameters, cancellation_token):
        self._check_cancelled(cancellation_token)
        if model_wrapper.id not in self.supported_model_ids:
            raise ValueError(
                f"El ataque '{self.id}' no es compatible con el modelo "
                f"'{model_wrapper.id}'"
            )
        if not model_wrapper.is_loaded():
            raise ValueError("Model is not loaded")
        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL Image")

        epsilon = float(parameters.epsilon)
        alpha = float(parameters.alpha)
        steps = int(parameters.steps)
        if not (0.0 <= epsilon <= 0.3 and 0.0 <= alpha <= 0.1 and 1 <= steps <= 100):
            raise ValueError("Invalid localization PGD parameters")

        clean = model_wrapper.prepare_detection_attack_input(image).detach()
        if clean.ndim != 4 or clean.shape[0] != 1 or clean.shape[1] != 3:
            raise ValueError("Detection attack input must have shape [1, 3, H, W]")
        if clean.shape[-2:] != (image.height, image.width):
            raise ValueError("Detection attack input must retain source geometry")
        if not torch.isfinite(clean).all() or clean.min() < 0 or clean.max() > 1:
            raise ValueError("Detection attack input must be finite in [0, 1]")

        adversarial = clean.clone()
        if epsilon > 0 and alpha > 0:
            # Inferencia SOLO una vez: se congelan cajas/clases originales.
            self._check_cancelled(cancellation_token)
            clean_detections = model_wrapper.detect(image)
            references = DetectionLocalizationReferences.from_detections(
                clean_detections,
                min_confidence=_REFERENCE_CONFIDENCE,
            )
            if (references.source_width, references.source_height) != image.size:
                raise ValueError("Localization references have mismatched geometry")

            # Sin objetos no existe un término L1/GIoU fundamentado. La imagen
            # queda intacta: nunca sintetizamos una referencia artificial.
            if references.objects:
                for _ in range(steps):
                    self._check_cancelled(cancellation_token)
                    candidate = adversarial.detach().requires_grad_(True)
                    with torch.enable_grad():
                        objective = model_wrapper.detection_localization_objective(
                            candidate, references
                        )
                        if not isinstance(objective, torch.Tensor) or objective.ndim != 0:
                            raise ValueError("Detection localization objective must be scalar")
                        if not torch.isfinite(objective):
                            raise ValueError("Detection localization objective must be finite")
                        (gradient,) = torch.autograd.grad(
                            objective, candidate, allow_unused=True
                        )
                    if gradient is None:
                        raise RuntimeError("Localization objective has no input gradient")
                    if not torch.isfinite(gradient).all():
                        raise ValueError("Localization input gradient must be finite")

                    # Ascenso sobre error de cajas, proyección L∞ y clipping RGB.
                    with torch.no_grad():
                        next_candidate = candidate + alpha * gradient.sign()
                        perturbation = (next_candidate - clean).clamp(-epsilon, epsilon)
                        adversarial = (clean + perturbation).clamp(0.0, 1.0).detach()

        self._check_cancelled(cancellation_token)
        # La salida real es PNG RGB de 8 bits. Restringimos TAMBIÉN el delta
        # cuantizado: el redondeo no puede exceder ε al guardar el resultado.
        original_pixels = torch.round(clean * 255.0).to(torch.int32)
        budget_in_levels = floor(epsilon * 255.0)
        adversarial_pixels = torch.round(adversarial * 255.0).to(torch.int32)
        adversarial_pixels = torch.maximum(
            torch.minimum(adversarial_pixels, original_pixels + budget_in_levels),
            original_pixels - budget_in_levels,
        ).clamp(0, 255).to(torch.uint8)
        final_tensor = adversarial_pixels.to(dtype=clean.dtype) / 255.0
        delta = final_tensor - clean
        perturbation_linf, perturbation_l2 = self._perturbation_norms(delta)
        return {
            "adversarial_image": to_pil_image(adversarial_pixels.squeeze(0).cpu()),
            "perturbation_image": perturbation_to_visualization(delta),
            "perturbation_linf": round(perturbation_linf, 4),
            "perturbation_l2": round(perturbation_l2, 4),
        }

import torch
from PIL import Image
from torchvision import transforms

from ml.attacks.base import AttackParameterSpec, BaseAdversarialAttack


class DetectionConfidenceAttack(BaseAdversarialAttack):
    id = "yolo_confidence"
    name = "Supresión de confianza"
    description = "Ataque de supresión de confianza para detección"
    knowledge_type = "white_box"
    revision = "detection-confidence-linf-v2"
    supported_task_types = ("detection",)
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
            description="Tamaño de paso de cada iteración",
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
            description="Número de iteraciones del ataque",
            value_type="integer",
            default=10,
            control="number",
            minimum=1,
            maximum=100,
            step=1,
        ),
    )

    def __init__(self):
        self.to_pil = transforms.ToPILImage()

    def apply(
        self,
        model_wrapper,
        image: Image.Image,
        parameters,
        original_class_index: int | None = None,
    ):
        """
        Ataque genérico de supresión de confianza para detección.

        Objetivo:
        Reducir el objetivo diferenciable de confianza que expone el detector
        activo. El ataque opera siempre en geometría fuente y delega en el modelo
        tanto el preprocesado diferenciable como la interpretación de su salida.

        Este ataque no usa etiquetas ground-truth. Es un ataque no dirigido de
        supresión de confianza.
        """
        if not model_wrapper.is_loaded():
            raise ValueError("Model is not loaded")

        epsilon = parameters.epsilon
        alpha = parameters.alpha
        steps = parameters.steps

        original_tensor = model_wrapper.prepare_detection_attack_input(image)
        adversarial_tensor = original_tensor.clone().detach()

        for _ in range(steps):
            adversarial_tensor.requires_grad = True

            with torch.enable_grad():
                confidence_loss = model_wrapper.detection_confidence_objective(
                    adversarial_tensor
                )

            if confidence_loss.ndim != 0:
                raise ValueError("Detection confidence objective must be a scalar")

            (input_gradient,) = torch.autograd.grad(
                confidence_loss,
                adversarial_tensor,
                allow_unused=True,
            )

            if input_gradient is None:
                raise RuntimeError(
                    "Detection confidence objective did not produce an input gradient"
                )

            gradient_sign = input_gradient.sign()

            # Descenso: buscamos reducir la confianza media de las detecciones.
            adversarial_tensor = adversarial_tensor - alpha * gradient_sign

            perturbation = torch.clamp(
                adversarial_tensor - original_tensor,
                min=-epsilon,
                max=epsilon,
            )

            adversarial_tensor = torch.clamp(
                original_tensor + perturbation,
                min=0,
                max=1,
            ).detach()

        perturbation = adversarial_tensor - original_tensor

        perturbation_linf, perturbation_l2 = self._perturbation_norms(perturbation)

        adversarial_image = self._tensor_to_image(adversarial_tensor)
        perturbation_image = self._perturbation_to_image(perturbation)

        return {
            "adversarial_image": adversarial_image,
            "perturbation_image": perturbation_image,
            "perturbation_linf": round(perturbation_linf, 4),
            "perturbation_l2": round(perturbation_l2, 4),
        }

    def _tensor_to_image(self, tensor: torch.Tensor) -> Image.Image:
        tensor = tensor.detach().to(device="cpu", dtype=torch.float32).squeeze(0)
        tensor = torch.clamp(tensor, 0, 1)

        return self.to_pil(tensor)

    def _perturbation_to_image(self, perturbation: torch.Tensor) -> Image.Image:
        perturbation = (
            perturbation.detach().to(device="cpu", dtype=torch.float32).squeeze(0)
        )
        perturbation = perturbation.abs()

        min_value = perturbation.min()
        max_value = perturbation.max()

        perturbation = (perturbation - min_value) / (max_value - min_value + 1e-8)

        return self.to_pil(perturbation)

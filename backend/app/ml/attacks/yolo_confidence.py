import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from ml.attacks.base import BaseAdversarialAttack


class YOLOConfidenceAttack(BaseAdversarialAttack):
    id = "yolo_confidence"
    name = "YOLO Confidence Attack"
    supported_task_types = ["detection"]

    def __init__(self):
        self.to_tensor = transforms.ToTensor()
        self.to_pil = transforms.ToPILImage()
        self.attack_size = 640

    def apply(self, model_wrapper, image: Image.Image, parameters):
        """
        Ataque específico para detección.

        Objetivo:
        Reducir las puntuaciones de confianza/clase que YOLO asigna a sus posibles
        detecciones. Para ello se hace descenso por gradiente sobre la media de las
        máximas puntuaciones de clase del tensor de salida bruto.

        Nota:
        Este ataque no usa etiquetas ground-truth. Es un ataque no dirigido de
        supresión de confianza.
        """
        if not model_wrapper.is_loaded():
            raise ValueError("Model is not loaded")

        epsilon = parameters.epsilon
        alpha = parameters.alpha
        steps = parameters.steps

        torch_model = model_wrapper.get_torch_model()
        torch_model.eval()

        original_tensor = self.to_tensor(image).unsqueeze(0)
        adversarial_tensor = original_tensor.clone().detach()

        for _ in range(steps):
            adversarial_tensor.requires_grad = True

            model_input = self._resize_for_yolo(adversarial_tensor)

            output = torch_model(model_input)
            prediction_tensor = self._extract_prediction_tensor(output)

            confidence_loss = self._confidence_objective(prediction_tensor)

            torch_model.zero_grad()
            confidence_loss.backward()

            gradient_sign = adversarial_tensor.grad.data.sign()

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

        perturbation_linf = torch.max(torch.abs(perturbation)).item()
        perturbation_l2 = torch.norm(perturbation).item()

        adversarial_image = self._tensor_to_image(adversarial_tensor)
        perturbation_image = self._perturbation_to_image(perturbation)

        return {
            "adversarial_image": adversarial_image,
            "perturbation_image": perturbation_image,
            "perturbation_linf": round(perturbation_linf, 4),
            "perturbation_l2": round(perturbation_l2, 4),
        }

    def _resize_for_yolo(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        YOLO espera tamaños compatibles con su arquitectura.
        Usamos 640x640 para el ataque.
        """
        return F.interpolate(
            tensor,
            size=(self.attack_size, self.attack_size),
            mode="bilinear",
            align_corners=False,
        )

    def _extract_prediction_tensor(self, output):
        """
        Ultralytics puede devolver tensores, listas o tuplas dependiendo del modo.
        Buscamos el tensor principal de predicción.
        """
        if isinstance(output, torch.Tensor):
            return output

        if isinstance(output, (list, tuple)):
            for item in output:
                if isinstance(item, torch.Tensor):
                    return item

                if isinstance(item, (list, tuple)):
                    for nested_item in item:
                        if isinstance(nested_item, torch.Tensor):
                            return nested_item

        raise ValueError("Could not extract YOLO prediction tensor from model output")

    def _confidence_objective(self, prediction_tensor: torch.Tensor) -> torch.Tensor:
        """
        YOLOv8 suele devolver predicciones con forma [B, C, N],
        donde las primeras posiciones corresponden a cajas y el resto a scores
        de clase.

        Tomamos las puntuaciones de clase y reducimos la media de la máxima
        puntuación por candidato.
        """
        if prediction_tensor.ndim != 3:
            raise ValueError(
                f"Unexpected YOLO prediction tensor shape: {prediction_tensor.shape}"
            )

        # Caso habitual YOLOv8: [B, 84, 8400]
        if prediction_tensor.shape[1] > prediction_tensor.shape[2]:
            # Formato alternativo tipo [B, N, C]
            class_scores = prediction_tensor[:, :, 4:]
        else:
            # Formato habitual [B, C, N]
            class_scores = prediction_tensor[:, 4:, :]

        if class_scores.numel() == 0:
            raise ValueError("YOLO prediction tensor does not contain class scores")

        max_class_scores = class_scores.max(dim=1).values

        return max_class_scores.mean()

    def _tensor_to_image(self, tensor: torch.Tensor) -> Image.Image:
        tensor = tensor.detach().cpu().squeeze(0)
        tensor = torch.clamp(tensor, 0, 1)

        return self.to_pil(tensor)

    def _perturbation_to_image(self, perturbation: torch.Tensor) -> Image.Image:
        perturbation = perturbation.detach().cpu().squeeze(0)
        perturbation = perturbation.abs()

        min_value = perturbation.min()
        max_value = perturbation.max()

        perturbation = (perturbation - min_value) / (max_value - min_value + 1e-8)

        return self.to_pil(perturbation)
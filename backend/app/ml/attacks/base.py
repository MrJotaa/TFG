from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from PIL import Image


@dataclass(frozen=True, slots=True)
class AdversarialAttackResult:
    """Resultado tensorial de un ataque en la entrada de clasificación."""

    clean_model_input: torch.Tensor
    adversarial_model_input: torch.Tensor
    model_space_delta: torch.Tensor

    adversarial_image: Image.Image
    perturbation_image: Image.Image

    perturbation_linf: float
    perturbation_l2: float


class BaseAdversarialAttack(ABC):
    """
    Clase base para ataques adversarios sobre imágenes.
    Define la interfaz común que deberán implementar todos los ataques.
    """

    id: str
    name: str
    supported_task_types: list[str] = []

    def supports_task_type(self, task_type: str) -> bool:
        return task_type in self.supported_task_types

    @abstractmethod
    def apply(
        self,
        model_wrapper,
        image: Image.Image,
        parameters,
        original_class_index: int | None = None,
    ):
        """
        Aplica el ataque adversario sobre una imagen.

        Args:
            model_wrapper: modelo activo del sistema.
            image: imagen original como objeto PIL.
            parameters: parámetros específicos del ataque.
            original_class_index: clase original ya calculada, cuando
                el servicio puede reutilizarla y evitar una inferencia.

        Returns:
            Resultado interno definido por el ataque concreto.
        """
        pass

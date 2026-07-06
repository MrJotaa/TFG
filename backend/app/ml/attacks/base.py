from abc import ABC, abstractmethod

from PIL import Image


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
            Imagen adversaria y métricas asociadas a la perturbación.
        """
        pass

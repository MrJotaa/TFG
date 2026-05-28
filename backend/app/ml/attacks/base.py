from abc import ABC, abstractmethod

from PIL import Image


class BaseAdversarialAttack(ABC):
    """
    Clase base para ataques adversarios sobre imágenes.
    Define la interfaz común que deberán implementar todos los ataques.
    """

    id: str
    name: str

    @abstractmethod
    def apply(
        self,
        model_wrapper,
        image: Image.Image,
        parameters,
    ):
        """
        Aplica el ataque adversario sobre una imagen.

        Args:
            model_wrapper: modelo activo del sistema.
            image: imagen original como objeto PIL.
            parameters: parámetros específicos del ataque.

        Returns:
            Imagen adversaria y métricas asociadas a la perturbación.
        """
        pass
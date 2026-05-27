from abc import ABC, abstractmethod
from PIL import Image

from schemas.predict import PredictResponse


class BaseImageClassificationModel(ABC):
    """
    Clase base para modelos de clasificación de imágenes.
    Define la interfaz común que deberán implementar todos los modelos.
    No se instancia directamente,
    sino que se extiende para crear implementaciones específicas de modelos.
    """

    id: str
    name: str

    @abstractmethod
    def load(self) -> None:
        """
        Carga el modelo y los recursos necesarios.
        """
        pass

    @abstractmethod
    def unload(self) -> None:
        """
        Libera los recursos asociados al modelo.
        """
        pass

    @abstractmethod
    def is_loaded(self) -> bool:
        """
        Indica si el modelo está cargado y listo para hacer predicciones.
        """
        pass

    @abstractmethod
    def preprocess(self, image: Image.Image):
        """
        Preprocesa la imagen de entrada según los requisitos del modelo.
        """
        pass

    @abstractmethod
    def predict(self, image: Image.Image) -> PredictResponse:
        """
        Ejecuta la inferencia sobre una imagen y devuelve una respuesta normalizada.
        """
        pass
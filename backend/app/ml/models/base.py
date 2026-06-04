from abc import ABC, abstractmethod
from PIL import Image

from schemas.detection import DetectionResponse
from schemas.predict import PredictResponse


class BaseModel(ABC):
    """
    Clase base común para cualquier modelo de IA usado por la aplicación.
    """

    id: str
    name: str
    task_type: str

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
        Indica si el modelo está cargado y listo para hacer inferencia.
        """
        pass


class BaseImageClassificationModel(BaseModel):
    """
    Clase base para modelos de clasificación de imágenes.
    """

    task_type = "classification"

    @abstractmethod
    def preprocess(self, image: Image.Image):
        """
        Preprocesa la imagen de entrada según los requisitos del modelo.
        """
        pass

    @abstractmethod
    def predict(self, image: Image.Image) -> PredictResponse:
        """
        Ejecuta la inferencia de clasificación sobre una imagen.
        """
        pass


class BaseObjectDetectionModel(BaseModel):
    """
    Clase base para modelos de detección de objetos.
    """

    task_type = "detection"

    @abstractmethod
    def detect(self, image: Image.Image) -> DetectionResponse:
        """
        Ejecuta detección de objetos sobre una imagen.
        """
        pass

    def get_annotated_image(self, image: Image.Image) -> Image.Image:
        """
        Devuelve una imagen anotada con las detecciones.

        Los modelos concretos pueden sobrescribirlo. Por defecto,
        no transforma la imagen.
        """
        return image
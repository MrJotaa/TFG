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

    def detect_with_visualization(
        self,
        image: Image.Image,
    ) -> tuple[DetectionResponse, Image.Image]:
        """
        Ejecuta detección y devuelve también una imagen visualmente anotada.

        Los modelos concretos pueden sobrescribirlo para evitar ejecutar
        inferencia dos veces.
        """
        detection_response = self.detect(image)
        annotated_image = self.get_annotated_image(image)

        return detection_response, annotated_image

    def get_torch_model(self):
        """
        Devuelve el modelo PyTorch interno si el detector lo expone.
        Necesario para técnicas de explainability como EigenCAM.
        """
        raise NotImplementedError(
            f"Model '{self.id}' does not expose a torch model"
        )

    def get_explainability_target_layer(self):
        """
        Devuelve la capa objetivo para técnicas CAM.
        """
        raise NotImplementedError(
            f"Model '{self.id}' does not define an explainability target layer"
        )
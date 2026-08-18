from abc import ABC, abstractmethod

import torch
from PIL import Image

from ml.detection_categories import DetectionClassCatalog
from ml.models.classification import (
    DetailedClassificationPrediction,
)
from ml.models.detection import DetectionResult
from ml.models.detection_inference import (
    DetectionEvaluationInferenceConfiguration,
)
from schemas.predict import PredictResponse


class BaseModel(ABC):
    """
    Clase base común para cualquier modelo de IA usado por la aplicación.
    """

    id: str
    name: str
    revision: str
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
    def prepare_attack_input(
        self,
        image: Image.Image,
    ) -> torch.Tensor:
        """
        Convierte la imagen al espacio de píxeles que consume el modelo.

        El tensor debe tener forma [1, C, H, W], valores en [0, 1]
        y estar situado en el mismo dispositivo que el modelo.
        """
        pass

    @abstractmethod
    def normalize_attack_input(
        self,
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        """
        Normaliza un tensor de píxeles preparado para el modelo.
        """
        pass

    @abstractmethod
    def preprocess(self, image: Image.Image):
        """
        Preprocesa la imagen de entrada según los requisitos del modelo.
        """
        pass

    @abstractmethod
    def predict_detailed(
        self,
        image: Image.Image,
        top_k: int = 5,
    ) -> DetailedClassificationPrediction:
        """
        Ejecuta una única inferencia y devuelve el vector completo
        de probabilidades junto con información de la predicción.
        """
        pass

    @abstractmethod
    def predict_detailed_from_prepared_input(
        self,
        prepared_input: torch.Tensor,
        top_k: int = 5,
    ) -> DetailedClassificationPrediction:
        """
        Predice desde un tensor float ya adaptado a la geometría del modelo.

        La implementación debe aplicar la normalización del modelo, pero no
        debe repetir transformaciones geométricas ni conversiones de imagen.
        """
        pass

    @abstractmethod
    def predict(self, image: Image.Image) -> PredictResponse:
        """
        Ejecuta la inferencia y construye la respuesta pública
        utilizada por el endpoint /predict.
        """
        pass

    @abstractmethod
    def predict_from_prepared_input(
        self,
        prepared_input: torch.Tensor,
    ) -> PredictResponse:
        """
        Construye la respuesta pública desde una entrada ya preparada.
        """
        pass

    @abstractmethod
    def get_class_labels(self) -> tuple[str, ...]:
        """
        Devuelve el catálogo canónico y ordenado de clases.

        El índice de cada etiqueta coincide con la posición de su
        probabilidad en predict_detailed().
        """
        pass


class BaseObjectDetectionModel(BaseModel):
    """
    Clase base para modelos de detección de objetos.
    """

    task_type = "detection"

    @abstractmethod
    def get_detection_class_catalog(self) -> DetectionClassCatalog:
        """
        Devuelve el catálogo completo de clases sin ejecutar inferencia.
        """
        pass

    @abstractmethod
    def detect(self, image: Image.Image) -> DetectionResult:
        """
        Ejecuta detección de objetos sobre una imagen.
        """
        pass

    def detect_for_evaluation(
        self,
        image: Image.Image,
        configuration: DetectionEvaluationInferenceConfiguration,
    ) -> DetectionResult:
        """
        Ejecuta una inferencia configurada para métricas basadas en scores.

        Los detectores concretos deben implementarla sin reutilizar
        silenciosamente los defaults interactivos.
        """
        raise NotImplementedError(
            f"Model '{self.id}' does not support configured detection evaluation"
        )

    def get_annotated_image(
        self,
        image: Image.Image,
    ) -> Image.Image:
        """
        Devuelve una imagen anotada con las detecciones.

        Los modelos concretos pueden sobrescribirlo. Por defecto,
        no transforma la imagen.
        """
        return image

    def detect_with_visualization(
        self,
        image: Image.Image,
    ) -> tuple[DetectionResult, Image.Image]:
        """
        Ejecuta detección y devuelve también una imagen anotada.

        Los modelos concretos pueden sobrescribirlo para evitar
        ejecutar inferencia dos veces.
        """
        detection_result = self.detect(image)
        annotated_image = self.get_annotated_image(image)

        return detection_result, annotated_image

    def get_torch_model(self):
        """
        Devuelve el modelo PyTorch interno si el detector lo expone.
        """
        raise NotImplementedError(f"Model '{self.id}' does not expose a torch model")

    def get_explainability_target_layer(self):
        """
        Devuelve la capa objetivo para técnicas CAM.
        """
        raise NotImplementedError(
            f"Model '{self.id}' does not define an explainability target layer"
        )

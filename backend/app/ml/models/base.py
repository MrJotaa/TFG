from abc import ABC, abstractmethod

import torch
from PIL import Image

from ml.detection_categories import DetectionClassCatalog
from ml.models.classification import (
    DetailedClassificationPrediction,
)
from ml.models.detection import DetectionResult
from ml.models.detection_localization import DetectionLocalizationReferences
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
    evaluation_attack_micro_batch_size = 1

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

    def get_torch_model(self) -> torch.nn.Module:
        """Devuelve el módulo PyTorch diferenciable usado por el clasificador."""
        raise NotImplementedError(f"Model '{self.id}' does not expose a torch model")

    def get_attack_input_shape(self) -> tuple[int, ...]:
        """Devuelve la forma CxHxW de una muestra en el dominio de ataque."""
        raise NotImplementedError(
            f"Model '{self.id}' does not expose its classification attack input shape"
        )

    def get_gradcam_target_layer(self) -> torch.nn.Module:
        """Devuelve la capa cuyas activaciones se explican con Grad-CAM."""
        raise NotImplementedError(
            f"Model '{self.id}' does not expose a Grad-CAM target layer"
        )

    def reshape_gradcam_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Adapta activaciones o gradientes de Grad-CAM al formato [B, C, H, W].

        Las CNN ya suelen exponer activaciones en ese formato y no necesitan
        transformación. Arquitecturas como Vision Transformers pueden
        sobrescribir este método para reconstruir su representación espacial.
        """
        return tensor

    def forward_attack_logits(
        self,
        prepared_input: torch.Tensor,
    ) -> torch.Tensor:
        """
        Ejecuta un forward diferenciable desde el dominio de ataque [0, 1].

        Los clasificadores cuyo forward nativo no devuelva directamente logits
        pueden sobrescribir este método sin modificar los ataques ni adaptadores.
        """
        if not self.is_loaded():
            raise ValueError("Model is not loaded")

        normalized_input = self.normalize_attack_input(prepared_input)
        return self.get_torch_model()(normalized_input)


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

    def prepare_detection_attack_input(
        self,
        image: Image.Image,
    ) -> torch.Tensor:
        """
        Prepara una imagen en geometría fuente para ataques diferenciables.

        El tensor debe tener forma [1, C, H, W], valores en [0, 1] y residir
        en el mismo dispositivo que el modelo. El preprocesado geométrico
        específico del detector se aplica dentro del objetivo diferenciable.
        """
        raise NotImplementedError(
            f"Model '{self.id}' does not support differentiable detection attacks"
        )

    def detection_confidence_objective(
        self,
        prepared_input: torch.Tensor,
    ) -> torch.Tensor:
        """
        Devuelve un objetivo escalar diferenciable de confianza de detección.

        La implementación concreta decide cómo preprocesar el tensor y qué
        salida interna del detector representa correctamente su confianza.
        """
        raise NotImplementedError(
            f"Model '{self.id}' does not expose a detection confidence objective"
        )

    def detection_localization_objective(
        self,
        prepared_input: torch.Tensor,
        references: DetectionLocalizationReferences,
    ) -> torch.Tensor:
        """Objetivo diferenciable de error de cajas con referencias fijas.

        ``references`` usa coordenadas xyxy de la imagen fuente e IDs de
        clase del modelo. No se redefine a partir de cada imagen perturbada.
        """
        raise NotImplementedError(
            f"Model '{self.id}' does not expose a detection localization objective"
        )

    def get_torch_model(self):
        """
        Devuelve el modelo PyTorch interno si el detector lo expone.
        """
        raise NotImplementedError(f"Model '{self.id}' does not expose a torch model")

    def get_detection_explainability_model(self) -> torch.nn.Module:
        """Devuelve el módulo PyTorch que ejecutará el motor CAM genérico."""
        model = self.get_torch_model()
        if not isinstance(model, torch.nn.Module):
            raise TypeError(
                f"Model '{self.id}' detection explainability model must be a "
                "torch module"
            )
        return model

    def reshape_detection_explainability_tensor(
        self,
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Adapta activaciones CAM de detección al formato espacial [B, C, H, W]."""
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("Detection explainability transform must return a tensor")
        if tensor.ndim != 4:
            raise ValueError(
                "Detection explainability activation must have shape [B, C, H, W] "
                f"after model transformation, got {tuple(tensor.shape)}"
            )
        return tensor

    def supports_detection_explainability(self) -> bool:
        """Indica si el detector expone actualmente el contrato CAM completo."""
        return False

    def prepare_detection_explainability_input(
        self,
        image: Image.Image,
    ) -> torch.Tensor:
        """Prepara la entrada científica utilizada por explainability de detección."""
        raise NotImplementedError(
            f"Model '{self.id}' does not support detection explainability input"
        )

    def project_detection_explainability_map(
        self,
        normalized_cam,
        source_image: Image.Image,
    ):
        """Proyecta un CAM desde geometría de modelo a geometría fuente."""
        raise NotImplementedError(
            f"Model '{self.id}' does not support detection CAM projection"
        )

    def get_explainability_target_layers(self):
        """Devuelve las capas objetivo para técnicas CAM de detección."""
        raise NotImplementedError(
            f"Model '{self.id}' does not define explainability target layers"
        )

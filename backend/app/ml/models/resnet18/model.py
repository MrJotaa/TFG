import gc

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

from ml.models.base import BaseImageClassificationModel
from ml.models.classification import (
    ClassificationClassScore,
    DetailedClassificationPrediction,
)
from schemas.predict import PredictResponse, TopKPrediction


class ResNet18Model(BaseImageClassificationModel):
    id = "resnet18"
    name = "ResNet18 pretrained"
    task_type = "classification"

    input_size = (224, 224)
    normalization_mean = (0.485, 0.456, 0.406)
    normalization_std = (0.229, 0.224, 0.225)

    def __init__(self):
        self.model = None
        self.categories = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._attack_input_transform = transforms.Compose(
            [
                transforms.Resize(self.input_size),
                transforms.ToTensor(),
            ]
        )

    def load(self) -> None:
        """
        Carga el modelo ResNet18 y el catálogo de ImageNet.
        """
        if self.is_loaded():
            return

        weights = models.ResNet18_Weights.DEFAULT

        self.model = models.resnet18(weights=weights)
        self.model.to(self.device)
        self.model.eval()

        self.categories = weights.meta["categories"]

    def unload(self) -> None:
        """
        Libera las referencias al modelo y sus categorías.
        """
        self.model = None
        self.categories = None

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def is_loaded(self) -> bool:
        return self.model is not None and self.categories is not None

    def prepare_attack_input(
        self,
        image: Image.Image,
    ) -> torch.Tensor:
        """
        Lleva la imagen al espacio de entrada real de ResNet18.

        Los ataques operan sobre este tensor sin normalizar para que
        epsilon mantenga su interpretación en el rango [0, 1].
        """
        rgb_image = image.convert("RGB")
        tensor = self._attack_input_transform(rgb_image)

        return tensor.unsqueeze(0).to(self.device)

    def normalize_attack_input(
        self,
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        """
        Aplica la normalización de ImageNet a un batch [B, C, H, W].
        """
        mean = torch.as_tensor(
            self.normalization_mean,
            dtype=tensor.dtype,
            device=tensor.device,
        ).view(1, -1, 1, 1)

        std = torch.as_tensor(
            self.normalization_std,
            dtype=tensor.dtype,
            device=tensor.device,
        ).view(1, -1, 1, 1)

        return (tensor - mean) / std

    def preprocess(
        self,
        image: Image.Image,
    ) -> torch.Tensor:
        """
        Prepara y normaliza una imagen para inferencia.
        """
        attack_input = self.prepare_attack_input(image)
        return self.normalize_attack_input(attack_input)

    def get_class_labels(self) -> tuple[str, ...]:
        """
        Devuelve el catálogo ordenado de clases de ImageNet.
        """
        if not self.is_loaded():
            raise ValueError("Model is not loaded")

        return tuple(self.categories)

    def predict_detailed(
        self,
        image: Image.Image,
        top_k: int = 5,
    ) -> DetailedClassificationPrediction:
        """
        Ejecuta una única inferencia y conserva todas las
        probabilidades sin redondear.
        """
        if not self.is_loaded():
            raise ValueError("Model is not loaded")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        normalized_input = self.preprocess(image)

        return self._predict_detailed_from_normalized_input(
            normalized_input=normalized_input,
            top_k=top_k,
        )

    def predict_detailed_from_prepared_input(
        self,
        prepared_input: torch.Tensor,
        top_k: int = 5,
    ) -> DetailedClassificationPrediction:
        """
        Predice desde un batch float ya adaptado a la geometría de ResNet18.

        El tensor permanece en el dominio de píxel [0, 1]. Este método aplica
        únicamente la normalización previa al forward.
        """
        if not self.is_loaded():
            raise ValueError("Model is not loaded")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        if prepared_input.ndim != 4:
            raise ValueError("Prepared input must have shape [N, C, H, W]")

        if not prepared_input.is_floating_point():
            raise ValueError("Prepared input must be a floating-point tensor")

        normalized_input = self.normalize_attack_input(prepared_input)

        return self._predict_detailed_from_normalized_input(
            normalized_input=normalized_input,
            top_k=top_k,
        )

    def _predict_detailed_from_normalized_input(
        self,
        normalized_input: torch.Tensor,
        top_k: int,
    ) -> DetailedClassificationPrediction:
        """Ejecuta el forward y comparte el postprocesado de clasificación."""

        with torch.inference_mode():
            output = self.model(normalized_input)
            probabilities = F.softmax(
                output[0],
                dim=0,
            )

        class_labels = self.get_class_labels()

        if probabilities.numel() != len(class_labels):
            raise ValueError("Model category count does not match probability count")

        top_k_count = min(
            top_k,
            probabilities.numel(),
        )

        top_probabilities, top_indices = torch.topk(
            probabilities,
            k=top_k_count,
        )

        class_scores = tuple(
            ClassificationClassScore(
                index=index.item(),
                label=class_labels[index.item()],
                probability=probability.item(),
            )
            for probability, index in zip(
                top_probabilities,
                top_indices,
            )
        )

        predicted_class = class_scores[0]

        prediction_margin = None

        if len(class_scores) >= 2:
            prediction_margin = (
                class_scores[0].probability - class_scores[1].probability
            )

        full_probabilities = tuple(probabilities.detach().cpu().tolist())

        return DetailedClassificationPrediction(
            class_index=predicted_class.index,
            class_label=predicted_class.label,
            confidence=predicted_class.probability,
            probabilities=full_probabilities,
            top_k=class_scores,
            prediction_margin=prediction_margin,
        )

    def _to_predict_response(
        self,
        detailed_prediction: DetailedClassificationPrediction,
    ) -> PredictResponse:
        """Convierte una predicción interna al contrato público existente."""
        top_k_predictions = [
            TopKPrediction(
                label=class_score.label,
                probability=round(
                    class_score.probability,
                    4,
                ),
            )
            for class_score in detailed_prediction.top_k
        ]

        return PredictResponse(
            prediction=detailed_prediction.class_label,
            confidence=round(
                detailed_prediction.confidence,
                4,
            ),
            top_k=top_k_predictions,
        )

    def predict(
        self,
        image: Image.Image,
    ) -> PredictResponse:
        """
        Construye la respuesta pública utilizada por /predict.

        Se mantiene el top-3 y el redondeo existentes para no
        modificar el contrato actual del frontend.
        """
        detailed_prediction = self.predict_detailed(
            image=image,
            top_k=3,
        )

        return self._to_predict_response(detailed_prediction)

    def predict_from_prepared_input(
        self,
        prepared_input: torch.Tensor,
    ) -> PredictResponse:
        """Construye la respuesta pública sin reprocesado geométrico."""
        detailed_prediction = self.predict_detailed_from_prepared_input(
            prepared_input=prepared_input,
            top_k=3,
        )

        return self._to_predict_response(detailed_prediction)

    def get_torch_model(self):
        if not self.is_loaded():
            raise ValueError("Model is not loaded")

        return self.model

    def get_gradcam_target_layer(self):
        if not self.is_loaded():
            raise ValueError("Model is not loaded")

        return self.model.layer4[-1]

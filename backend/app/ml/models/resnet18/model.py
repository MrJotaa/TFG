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

    def __init__(self):
        self.model = None
        self.categories = None

    def load(self) -> None:
        """
        Carga el modelo ResNet18 y el catálogo de ImageNet.
        """
        if self.is_loaded():
            return

        weights = models.ResNet18_Weights.DEFAULT

        self.model = models.resnet18(weights=weights)
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

    def preprocess(
        self,
        image: Image.Image,
    ) -> torch.Tensor:
        """
        Preprocesa una imagen usando la normalización de ImageNet.
        """
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        tensor = transform(image)

        return tensor.unsqueeze(0)

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

        input_tensor = self.preprocess(image)

        with torch.no_grad():
            output = self.model(input_tensor)
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

    def get_torch_model(self):
        if not self.is_loaded():
            raise ValueError("Model is not loaded")

        return self.model

    def get_gradcam_target_layer(self):
        if not self.is_loaded():
            raise ValueError("Model is not loaded")

        return self.model.layer4[-1]

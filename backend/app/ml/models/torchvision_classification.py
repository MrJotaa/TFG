import gc
from abc import abstractmethod
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from ml.models.base import BaseImageClassificationModel
from ml.models.classification import (
    ClassificationClassScore,
    DetailedClassificationPrediction,
)
from schemas.predict import PredictResponse, TopKPrediction


class TorchvisionImageClassificationModel(BaseImageClassificationModel):
    """
    Base reutilizable para clasificadores de imágenes implementados con Torchvision.

    Mantiene el dominio de ataque en píxeles [0, 1] separado de la
    normalización del modelo y centraliza el ciclo de vida y el postprocesado
    común. Las arquitecturas concretas solo deben construir su módulo PyTorch
    y el catálogo de clases asociado a los mismos pesos.
    """

    input_size: tuple[int, int]
    normalization_mean: tuple[float, ...]
    normalization_std: tuple[float, ...]

    public_top_k = 3

    def __init__(self) -> None:
        self.model: torch.nn.Module | None = None
        self.categories: Sequence[str] | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._attack_input_transform = transforms.Compose(
            [
                transforms.Resize(self.input_size),
                transforms.ToTensor(),
            ]
        )

    @abstractmethod
    def _build_model(self) -> tuple[torch.nn.Module, Sequence[str]]:
        """
        Construye el modelo y devuelve su catálogo ordenado de clases.

        Ambos elementos deben proceder de la misma revisión de pesos para
        mantener alineados los índices de los logits y las etiquetas.
        """
        pass

    def load(self) -> None:
        """Carga el modelo, lo mueve al dispositivo de ejecución y activa eval."""
        if self.is_loaded():
            return

        model, categories = self._build_model()
        model.to(self.device)
        model.eval()

        self.model = model
        self.categories = tuple(categories)

    def unload(self) -> None:
        """Libera las referencias compartidas y la memoria de acelerador."""
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
        Lleva una imagen RGB al dominio de ataque del clasificador.

        El tensor conserva valores [0, 1] para que parámetros como epsilon
        mantengan una interpretación independiente de la normalización.
        """
        rgb_image = image.convert("RGB")
        tensor = self._attack_input_transform(rgb_image)

        return tensor.unsqueeze(0).to(self.device)

    def normalize_attack_input(
        self,
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Aplica la normalización declarada por el clasificador."""
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
        """Prepara la geometría de entrada y aplica la normalización."""
        attack_input = self.prepare_attack_input(image)
        return self.normalize_attack_input(attack_input)

    def get_class_labels(self) -> tuple[str, ...]:
        """Devuelve el catálogo ordenado asociado a los logits del modelo."""
        if not self.is_loaded():
            raise ValueError("Model is not loaded")

        return tuple(self.categories)

    def predict_detailed(
        self,
        image: Image.Image,
        top_k: int = 5,
    ) -> DetailedClassificationPrediction:
        """Ejecuta inferencia y conserva el vector completo de probabilidades."""
        self._validate_prediction_request(top_k=top_k)

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
        Predice desde un tensor ya adaptado a la geometría del clasificador.

        El tensor permanece en [0, 1]; aquí solo se aplica la normalización
        antes del forward, sin repetir transformaciones geométricas.
        """
        self._validate_prediction_request(top_k=top_k)

        if prepared_input.ndim != 4:
            raise ValueError("Prepared input must have shape [N, C, H, W]")

        if not prepared_input.is_floating_point():
            raise ValueError("Prepared input must be a floating-point tensor")

        normalized_input = self.normalize_attack_input(prepared_input)

        return self._predict_detailed_from_normalized_input(
            normalized_input=normalized_input,
            top_k=top_k,
        )

    def _validate_prediction_request(self, *, top_k: int) -> None:
        if not self.is_loaded():
            raise ValueError("Model is not loaded")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

    def _predict_detailed_from_normalized_input(
        self,
        normalized_input: torch.Tensor,
        top_k: int,
    ) -> DetailedClassificationPrediction:
        """Ejecuta el forward y comparte el postprocesado de clasificación."""
        with torch.inference_mode():
            output = self.get_torch_model()(normalized_input)
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
        """Convierte la predicción interna al contrato HTTP existente."""
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
        """Construye la respuesta pública utilizada por /predict."""
        detailed_prediction = self.predict_detailed(
            image=image,
            top_k=self.public_top_k,
        )

        return self._to_predict_response(detailed_prediction)

    def predict_from_prepared_input(
        self,
        prepared_input: torch.Tensor,
    ) -> PredictResponse:
        """Construye la respuesta pública sin reprocesado geométrico."""
        detailed_prediction = self.predict_detailed_from_prepared_input(
            prepared_input=prepared_input,
            top_k=self.public_top_k,
        )

        return self._to_predict_response(detailed_prediction)

    def get_torch_model(self) -> torch.nn.Module:
        if not self.is_loaded():
            raise ValueError("Model is not loaded")

        return self.model

    def get_attack_input_shape(self) -> tuple[int, ...]:
        return (3, *self.input_size)

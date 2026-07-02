import gc

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

from ml.models.base import BaseImageClassificationModel
from schemas.predict import PredictResponse, TopKPrediction


class ResNet18Model(BaseImageClassificationModel):
    id = "resnet18"
    name = "ResNet18 pretrained"
    task_type = "classification"

    def __init__(self):
        self.model = None
        self.categories = None

    def load(self) -> None:  # Carga el modelo y los recursos necesarios.
        # Si ya está cargado, no lo volvemos a cargar.
        if self.is_loaded():
            return

        weights = models.ResNet18_Weights.DEFAULT
        self.model = models.resnet18(weights=weights)
        self.model.eval()
        self.categories = weights.meta["categories"]

    def unload(self) -> None:
        # Libera referencias al modelo y categorías.
        self.model = None
        self.categories = None

        # Fuerza recolección de basura en CPU.
        gc.collect()

        # Si en el futuro se usa GPU, limpia caché CUDA.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def is_loaded(self) -> bool:
        return self.model is not None and self.categories is not None

    def preprocess(
        self, image: Image.Image
    ):  # Preprocesa la imagen de entrada según los requisitos del modelo.
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
        tensor = tensor.unsqueeze(0)
        return tensor

    def predict(self, image: Image.Image) -> PredictResponse:
        """
        Ejecuta la inferencia sobre una imagen y devuelve una respuesta normalizada.
        """
        if not self.is_loaded():
            raise ValueError("Model is not loaded")

        input_tensor = self.preprocess(image)

        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = F.softmax(
                output[0], dim=0
            )  # Convierte las salidas del modelo en probabilidades utilizando softmax

        top_probs, top_indices = torch.topk(probabilities, k=3)

        top_k_predictions = [
            TopKPrediction(
                label=self.categories[idx.item()],
                probability=round(prob.item(), 4),
            )
            for prob, idx in zip(top_probs, top_indices)
        ]

        return PredictResponse(
            prediction=top_k_predictions[0].label,
            confidence=top_k_predictions[0].probability,
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

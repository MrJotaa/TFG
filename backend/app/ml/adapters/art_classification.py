from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from art.estimators.classification import PyTorchClassifier

if TYPE_CHECKING:
    from ml.models.base import BaseImageClassificationModel

ART_CLASSIFICATION_CLIP_VALUES = (0.0, 1.0)


class _ARTClassificationLogitsModule(torch.nn.Module):
    """Expone a ART el forward diferenciable definido por el modelo de la app."""

    def __init__(
        self,
        model_wrapper: BaseImageClassificationModel,
        torch_model: torch.nn.Module,
        nb_classes: int,
    ) -> None:
        super().__init__()
        self.model = torch_model
        self._model_wrapper = model_wrapper
        self._nb_classes = nb_classes

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = self._model_wrapper.forward_attack_logits(inputs)

        if not isinstance(logits, torch.Tensor):
            raise TypeError("Classification attack forward must return a torch.Tensor")
        if logits.ndim != 2:
            raise ValueError(
                "Classification attack forward must return logits with shape "
                "[batch, classes]"
            )
        if logits.shape[1] != self._nb_classes:
            raise ValueError(
                "Classification attack forward class count does not match the "
                "model class catalog"
            )

        return logits


class ARTClassificationAdapter:
    """Adapta un clasificador PyTorch interno al contrato de clasificación de ART.

    ART opera sobre el mismo dominio de píxel que los ataques de la aplicación:
    tensores NCHW float en [0, 1]. La normalización y cualquier transformación
    diferenciable específica del clasificador permanecen bajo responsabilidad del
    propio modelo mediante ``forward_attack_logits``.
    """

    def __init__(self, model_wrapper: BaseImageClassificationModel) -> None:
        if getattr(model_wrapper, "task_type", None) != "classification":
            raise ValueError(
                "ART classification adapter requires a classification model"
            )
        if not model_wrapper.is_loaded():
            raise ValueError("Model is not loaded")

        torch_model = model_wrapper.get_torch_model()
        if not isinstance(torch_model, torch.nn.Module):
            raise TypeError("Classification model must expose a torch.nn.Module")

        input_shape = tuple(model_wrapper.get_attack_input_shape())
        self._validate_input_shape(input_shape)

        class_labels = tuple(model_wrapper.get_class_labels())
        if len(class_labels) < 2:
            raise ValueError("Classification model must expose at least two classes")

        device = self._get_model_device(torch_model)
        if device.type not in {"cpu", "cuda"}:
            raise ValueError(
                f"ART PyTorchClassifier does not support device type '{device.type}'"
            )

        self.model_wrapper = model_wrapper
        self.torch_model = torch_model
        self.input_shape = input_shape
        self.nb_classes = len(class_labels)
        self.device = device
        self.clip_values = ART_CLASSIFICATION_CLIP_VALUES

        logits_model = _ARTClassificationLogitsModule(
            model_wrapper=model_wrapper,
            torch_model=torch_model,
            nb_classes=self.nb_classes,
        )

        self.estimator = self._build_estimator(logits_model)

    def to_art_input(self, prepared_input: torch.Tensor) -> np.ndarray:
        """Convierte una entrada preparada por la app al formato NumPy de ART."""
        self._validate_prepared_tensor(prepared_input)

        art_input = (
            prepared_input.detach()
            .to(device="cpu", dtype=torch.float32)
            .contiguous()
            .numpy()
        )
        minimum, maximum = self.clip_values

        if not np.isfinite(art_input).all():
            raise ValueError("Prepared attack input must contain only finite values")
        if np.any(art_input < minimum) or np.any(art_input > maximum):
            raise ValueError("Prepared attack input must contain values in [0, 1]")

        return art_input

    def to_model_input(self, art_input: np.ndarray) -> torch.Tensor:
        """Convierte la salida NumPy de ART de nuevo al tensor de ataque de la app."""
        array = np.asarray(art_input, dtype=np.float32)
        expected_ndim = len(self.input_shape) + 1

        if not np.isfinite(array).all():
            raise ValueError("ART output must contain only finite values")
        if array.ndim != expected_ndim:
            raise ValueError(
                f"ART input must have {expected_ndim} dimensions including batch"
            )
        if tuple(array.shape[1:]) != self.input_shape:
            raise ValueError(
                f"ART input shape {tuple(array.shape[1:])} does not match "
                f"expected shape {self.input_shape}"
            )

        tensor = torch.from_numpy(np.ascontiguousarray(array)).to(self.device)
        minimum, maximum = self.clip_values
        return torch.clamp(tensor, min=minimum, max=maximum)

    def _build_estimator(
        self,
        logits_model: torch.nn.Module,
    ) -> PyTorchClassifier:
        estimator_kwargs = {
            "model": logits_model,
            "loss": torch.nn.CrossEntropyLoss(),
            "input_shape": self.input_shape,
            "nb_classes": self.nb_classes,
            "optimizer": None,
            "channels_first": True,
            "clip_values": self.clip_values,
            "device_type": "gpu" if self.device.type == "cuda" else "cpu",
        }

        if self.device.type != "cuda":
            return PyTorchClassifier(**estimator_kwargs)

        cuda_index = self.device.index
        if cuda_index is None:
            cuda_index = torch.cuda.current_device()

        # ART elige torch.cuda.current_device() cuando device_type="gpu". El
        # contexto garantiza que no mueva inadvertidamente el modelo a otra GPU.
        with torch.cuda.device(cuda_index):
            return PyTorchClassifier(**estimator_kwargs)

    def _validate_prepared_tensor(self, prepared_input: torch.Tensor) -> None:
        if not isinstance(prepared_input, torch.Tensor):
            raise TypeError("Prepared attack input must be a torch.Tensor")
        if not prepared_input.is_floating_point():
            raise ValueError("Prepared attack input must be floating point")

        expected_ndim = len(self.input_shape) + 1
        if prepared_input.ndim != expected_ndim:
            raise ValueError(
                f"Prepared attack input must have {expected_ndim} dimensions "
                "including batch"
            )
        if tuple(prepared_input.shape[1:]) != self.input_shape:
            raise ValueError(
                f"Prepared attack input shape {tuple(prepared_input.shape[1:])} "
                f"does not match expected shape {self.input_shape}"
            )

    @staticmethod
    def _validate_input_shape(input_shape: tuple[int, ...]) -> None:
        if not input_shape or any(
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension <= 0
            for dimension in input_shape
        ):
            raise ValueError(
                "Attack input shape must contain positive integer dimensions"
            )

    @staticmethod
    def _get_model_device(torch_model: torch.nn.Module) -> torch.device:
        try:
            return next(torch_model.parameters()).device
        except StopIteration:
            try:
                return next(torch_model.buffers()).device
            except StopIteration:
                return torch.device("cpu")

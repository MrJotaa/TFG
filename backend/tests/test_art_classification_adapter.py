import numpy as np
import pytest
import torch

from core.cancellation import CancellationToken, OperationCancelled
from ml.adapters.art_classification import ARTClassificationAdapter


class TinyClassifier(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        score = inputs.mean(dim=(1, 2, 3)) * self.scale
        return torch.stack((score, -score), dim=1)


class FakeClassificationModel:
    id = "tiny-classifier"
    task_type = "classification"

    def __init__(self, *, loaded: bool = True) -> None:
        self.model = TinyClassifier()
        self.loaded = loaded
        self.normalization_calls = 0

    def is_loaded(self) -> bool:
        return self.loaded

    def get_torch_model(self) -> torch.nn.Module:
        return self.model

    def get_attack_input_shape(self) -> tuple[int, ...]:
        return (3, 4, 4)

    def get_class_labels(self) -> tuple[str, ...]:
        return ("positive", "negative")

    def normalize_attack_input(self, tensor: torch.Tensor) -> torch.Tensor:
        self.normalization_calls += 1
        return (tensor - 0.5) / 0.25

    def forward_attack_logits(self, prepared_input: torch.Tensor) -> torch.Tensor:
        return self.model(self.normalize_attack_input(prepared_input))


class FakeDetectionModel(FakeClassificationModel):
    task_type = "detection"


def test_art_adapter_exposes_expected_estimator_contract():
    model = FakeClassificationModel()

    adapter = ARTClassificationAdapter(model)

    assert adapter.input_shape == (3, 4, 4)
    assert adapter.nb_classes == 2
    assert adapter.clip_values == (0.0, 1.0)
    assert adapter.estimator.input_shape == (3, 4, 4)
    assert adapter.estimator.nb_classes == 2
    assert tuple(adapter.estimator.clip_values) == (0.0, 1.0)
    assert adapter.estimator.channels_first is True
    assert adapter.estimator.optimizer is None
    assert adapter.estimator.device == adapter.device


def test_art_estimator_predicts_logits_through_model_normalization():
    model = FakeClassificationModel()
    adapter = ARTClassificationAdapter(model)
    art_input = np.full((1, 3, 4, 4), 0.75, dtype=np.float32)

    predictions = adapter.estimator.predict(art_input)

    assert predictions.shape == (1, 2)
    assert predictions[0, 0] == pytest.approx(1.0)
    assert predictions[0, 1] == pytest.approx(-1.0)
    assert model.normalization_calls == 1


def test_art_estimator_exposes_loss_gradients_in_pixel_space():
    model = FakeClassificationModel()
    adapter = ARTClassificationAdapter(model)
    art_input = np.full((1, 3, 4, 4), 0.75, dtype=np.float32)
    labels = np.array([[1.0, 0.0]], dtype=np.float32)
    gradients = adapter.estimator.loss_gradient(art_input, labels)

    assert gradients.shape == art_input.shape
    assert np.isfinite(gradients).all()
    assert not np.allclose(gradients, 0.0)


def test_art_estimator_exposes_class_gradients_for_white_box_attacks():
    model = FakeClassificationModel()
    adapter = ARTClassificationAdapter(model)
    art_input = np.full((1, 3, 4, 4), 0.75, dtype=np.float32)

    gradients = adapter.estimator.class_gradient(art_input)

    assert gradients.shape == (1, 2, 3, 4, 4)
    assert np.isfinite(gradients).all()
    assert not np.allclose(gradients, 0.0)


def test_art_adapter_converts_between_model_and_art_inputs():
    model = FakeClassificationModel()
    adapter = ARTClassificationAdapter(model)
    prepared = torch.full((2, 3, 4, 4), 0.25, dtype=torch.float32)

    art_input = adapter.to_art_input(prepared)
    restored = adapter.to_model_input(art_input)

    assert art_input.dtype == np.float32
    assert art_input.shape == (2, 3, 4, 4)
    assert restored.device == adapter.device
    assert restored.dtype == torch.float32
    assert torch.allclose(restored.cpu(), prepared)


def test_art_adapter_clamps_generated_values_back_to_model_domain():
    model = FakeClassificationModel()
    adapter = ARTClassificationAdapter(model)
    art_input = np.array(
        [[[[1.1] * 4] * 4] * 3, [[[-0.1] * 4] * 4] * 3],
        dtype=np.float32,
    )

    restored = adapter.to_model_input(art_input)

    assert restored.min().item() == pytest.approx(0.0)
    assert restored.max().item() == pytest.approx(1.0)


def test_art_adapter_rejects_non_classification_models():
    _prepared_arg_1 = FakeDetectionModel()

    with pytest.raises(
        ValueError,
        match="requires a classification model",
    ):
        ARTClassificationAdapter(_prepared_arg_1)


def test_art_adapter_rejects_unloaded_models():
    _prepared_arg_1_2 = FakeClassificationModel(loaded=False)

    with pytest.raises(ValueError, match="Model is not loaded"):
        ARTClassificationAdapter(_prepared_arg_1_2)


def test_art_adapter_rejects_prepared_input_with_wrong_shape():
    adapter = ARTClassificationAdapter(FakeClassificationModel())

    _prepared_arg_1_3 = torch.zeros((1, 3, 8, 8), dtype=torch.float32)

    with pytest.raises(ValueError, match="does not match expected shape"):
        adapter.to_art_input(_prepared_arg_1_3)


def test_art_estimator_honours_cooperative_cancellation():
    model = FakeClassificationModel()
    cancellation_token = CancellationToken()
    adapter = ARTClassificationAdapter(
        model,
        cancellation_token=cancellation_token,
    )
    art_input = np.full((1, 3, 4, 4), 0.75, dtype=np.float32)

    cancellation_token.cancel()

    with pytest.raises(OperationCancelled, match="Operation cancelled"):
        adapter.estimator.predict(art_input)


def test_art_estimator_normalizes_float64_inputs_to_model_dtype():
    model = FakeClassificationModel()
    adapter = ARTClassificationAdapter(model)
    art_input = np.full((1, 3, 4, 4), 0.75, dtype=np.float64)

    predictions = adapter.estimator.predict(art_input)

    assert predictions.shape == (1, 2)
    assert np.isfinite(predictions).all()

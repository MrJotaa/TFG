import pytest
import torch
from PIL import Image

from ml.models.torchvision_classification import TorchvisionImageClassificationModel


class FixedLogitModel(torch.nn.Module):
    def __init__(self, logits: tuple[float, ...]):
        super().__init__()
        self.register_buffer("logits", torch.tensor(logits, dtype=torch.float32))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.logits.unsqueeze(0).expand(inputs.shape[0], -1)


class FakeTorchvisionClassifier(TorchvisionImageClassificationModel):
    id = "fake-torchvision-classifier"
    name = "Fake Torchvision classifier"
    revision = "test-v1"

    input_size = (8, 6)
    normalization_mean = (0.5, 0.5, 0.5)
    normalization_std = (0.25, 0.25, 0.25)

    def __init__(self) -> None:
        self.build_calls = 0
        self.built_model = FixedLogitModel((0.1, 0.4, -0.2))
        super().__init__()

    def _build_model(self) -> tuple[torch.nn.Module, tuple[str, ...]]:
        self.build_calls += 1
        return self.built_model, ("class-0", "class-1", "class-2")


def test_torchvision_classifier_centralizes_model_lifecycle():
    classifier = FakeTorchvisionClassifier()

    assert classifier.is_loaded() is False

    classifier.load()

    assert classifier.is_loaded() is True
    assert classifier.build_calls == 1
    assert classifier.get_torch_model() is classifier.built_model
    assert classifier.get_class_labels() == ("class-0", "class-1", "class-2")
    assert classifier.built_model.training is False

    classifier.load()

    assert classifier.build_calls == 1

    classifier.unload()

    assert classifier.is_loaded() is False


def test_torchvision_classifier_prepares_rgb_attack_domain_at_declared_resolution():
    classifier = FakeTorchvisionClassifier()
    image = Image.new("L", (31, 17), color=128)

    prepared_input = classifier.prepare_attack_input(image)

    assert prepared_input.shape == (1, 3, 8, 6)
    assert prepared_input.device.type == classifier.device.type
    assert prepared_input.is_floating_point()
    assert prepared_input.min().item() >= 0.0
    assert prepared_input.max().item() <= 1.0
    assert classifier.get_attack_input_shape() == (3, 8, 6)


def test_torchvision_classifier_normalization_is_separate_from_attack_domain():
    classifier = FakeTorchvisionClassifier()
    prepared_input = torch.full((1, 3, 8, 6), 0.75, dtype=torch.float32)

    normalized_input = classifier.normalize_attack_input(prepared_input)

    assert torch.allclose(normalized_input, torch.ones_like(prepared_input))
    assert torch.equal(prepared_input, torch.full_like(prepared_input, 0.75))


def test_torchvision_classifier_shares_prediction_postprocessing():
    classifier = FakeTorchvisionClassifier()
    classifier.load()

    prediction = classifier.predict_detailed_from_prepared_input(
        torch.zeros((1, 3, 8, 6), dtype=torch.float32),
        top_k=2,
    )
    response = classifier.predict_from_prepared_input(
        torch.zeros((1, 3, 8, 6), dtype=torch.float32)
    )

    expected_probabilities = torch.softmax(
        torch.tensor((0.1, 0.4, -0.2), dtype=torch.float32),
        dim=0,
    )

    assert prediction.class_index == 1
    assert prediction.class_label == "class-1"
    assert prediction.probabilities == pytest.approx(expected_probabilities.tolist())
    assert [score.index for score in prediction.top_k] == [1, 0]
    assert response.prediction == "class-1"
    assert len(response.top_k) == 3
    assert response.confidence == round(expected_probabilities[1].item(), 4)

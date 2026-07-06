import pytest
import torch
from PIL import Image

from ml.models.resnet18.model import ResNet18Model


class FixedLogitModel(torch.nn.Module):
    """
    Modelo sintético que devuelve siempre los mismos logits.
    """

    def __init__(self, logits: list[float]):
        super().__init__()

        self.register_buffer(
            "_logits",
            torch.tensor(
                logits,
                dtype=torch.float32,
            ),
        )

        self.forward_calls = 0

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        self.forward_calls += 1

        batch_size = inputs.shape[0]

        return self._logits.unsqueeze(0).expand(
            batch_size,
            -1,
        )


def _create_image() -> Image.Image:
    return Image.new(
        "RGB",
        (8, 8),
        color=(128, 128, 128),
    )


def _create_loaded_model(
    logits: list[float],
) -> tuple[ResNet18Model, FixedLogitModel]:
    torch_model = FixedLogitModel(logits)

    model = ResNet18Model()
    model.model = torch_model
    model.categories = [f"class-{index}" for index in range(len(logits))]

    model.preprocess = lambda image: torch.zeros(
        1,
        3,
        2,
        2,
    )

    return model, torch_model


def test_predict_detailed_returns_full_probability_vector():
    logits = [
        0.123456,
        1.234567,
        -0.456789,
        0.765432,
    ]

    model, torch_model = _create_loaded_model(logits)

    prediction = model.predict_detailed(
        image=_create_image(),
        top_k=3,
    )

    expected_probabilities = torch.softmax(
        torch.tensor(logits),
        dim=0,
    )

    expected_top_probabilities, expected_top_indices = torch.topk(
        expected_probabilities,
        k=3,
    )

    assert prediction.class_index == (expected_top_indices[0].item())
    assert prediction.class_label == (f"class-{expected_top_indices[0].item()}")

    assert prediction.confidence == pytest.approx(expected_top_probabilities[0].item())

    assert prediction.probabilities == pytest.approx(expected_probabilities.tolist())

    assert len(prediction.probabilities) == len(logits)
    assert sum(prediction.probabilities) == pytest.approx(1.0)

    assert [
        class_score.index for class_score in prediction.top_k
    ] == expected_top_indices.tolist()

    assert [
        class_score.probability for class_score in prediction.top_k
    ] == pytest.approx(expected_top_probabilities.tolist())

    expected_margin = (
        expected_top_probabilities[0] - expected_top_probabilities[1]
    ).item()

    assert prediction.prediction_margin == pytest.approx(expected_margin)

    assert torch_model.forward_calls == 1


def test_predict_preserves_public_top_three_contract():
    logits = [
        0.123456,
        1.234567,
        -0.456789,
        0.765432,
    ]

    model, torch_model = _create_loaded_model(logits)

    response = model.predict(_create_image())

    expected_probabilities = torch.softmax(
        torch.tensor(logits),
        dim=0,
    )

    expected_top_probabilities, expected_top_indices = torch.topk(
        expected_probabilities,
        k=3,
    )

    assert response.prediction == (f"class-{expected_top_indices[0].item()}")
    assert response.confidence == round(
        expected_top_probabilities[0].item(),
        4,
    )

    assert len(response.top_k) == 3

    for position, top_prediction in enumerate(response.top_k):
        expected_index = expected_top_indices[position].item()

        assert top_prediction.label == (f"class-{expected_index}")
        assert top_prediction.probability == round(
            expected_top_probabilities[position].item(),
            4,
        )

    assert not hasattr(response, "probabilities")
    assert torch_model.forward_calls == 1


def test_predict_detailed_respects_requested_top_k():
    model, _ = _create_loaded_model([0.1, 0.2, 0.3, 0.4])

    prediction = model.predict_detailed(
        image=_create_image(),
        top_k=2,
    )

    assert len(prediction.top_k) == 2


def test_predict_detailed_limits_top_k_to_class_count():
    model, _ = _create_loaded_model([0.1, 0.2, 0.3, 0.4])

    prediction = model.predict_detailed(
        image=_create_image(),
        top_k=20,
    )

    assert len(prediction.top_k) == 4


def test_predict_detailed_rejects_invalid_top_k():
    model, torch_model = _create_loaded_model([0.1, 0.2])

    with pytest.raises(
        ValueError,
        match="top_k must be greater than zero",
    ):
        model.predict_detailed(
            image=_create_image(),
            top_k=0,
        )

    assert torch_model.forward_calls == 0


def test_get_class_labels_returns_ordered_catalog():
    model, _ = _create_loaded_model([0.1, 0.2, 0.3])

    assert model.get_class_labels() == (
        "class-0",
        "class-1",
        "class-2",
    )


def test_prediction_requires_loaded_model():
    model = ResNet18Model()

    with pytest.raises(
        ValueError,
        match="Model is not loaded",
    ):
        model.predict_detailed(_create_image())

    with pytest.raises(
        ValueError,
        match="Model is not loaded",
    ):
        model.predict(_create_image())

    with pytest.raises(
        ValueError,
        match="Model is not loaded",
    ):
        model.get_class_labels()


def test_predict_detailed_validates_category_count():
    model, _ = _create_loaded_model([0.1, 0.2, 0.3])

    model.categories = [
        "class-0",
        "class-1",
    ]

    with pytest.raises(
        ValueError,
        match=("Model category count does not match probability count"),
    ):
        model.predict_detailed(_create_image())


def test_single_class_prediction_has_no_margin():
    model, _ = _create_loaded_model([1.0])

    prediction = model.predict_detailed(
        image=_create_image(),
        top_k=1,
    )

    assert prediction.class_index == 0
    assert prediction.prediction_margin is None


def test_resnet_attack_input_uses_model_resolution():
    model = ResNet18Model()

    attack_input = model.prepare_attack_input(Image.new("RGB", (1920, 1279)))

    assert attack_input.shape == (1, 3, 224, 224)
    assert attack_input.device.type == model.device.type
    assert attack_input.min().item() >= 0.0
    assert attack_input.max().item() <= 1.0


def test_resnet_preprocess_reuses_attack_input_contract():
    model = ResNet18Model()
    image = Image.new("RGB", (1280, 720), color=(128, 128, 128))

    attack_input = model.prepare_attack_input(image)
    preprocessed = model.preprocess(image)

    assert preprocessed.shape == attack_input.shape
    assert torch.allclose(
        preprocessed,
        model.normalize_attack_input(attack_input),
    )

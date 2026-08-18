from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from ml.attacks.yolo_confidence import YOLOConfidenceAttack


class TinyYOLOModel(torch.nn.Module):
    """
    Detector sintético con salida en formato YOLO [B, C, N].

    Los cuatro primeros canales representan las cajas.
    Los dos últimos representan puntuaciones de clase.

    La confianza depende directamente de la media de la imagen.
    Por tanto, el descenso aplicado por el ataque debe reducir
    tanto los valores de entrada como la confianza resultante.
    """

    def __init__(self):
        super().__init__()
        self.forward_calls = 0

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.forward_calls += 1

        batch_size = inputs.shape[0]
        candidate_count = 10

        score = inputs.mean(dim=(1, 2, 3))

        boxes = torch.zeros(
            batch_size,
            4,
            candidate_count,
            device=inputs.device,
            dtype=inputs.dtype,
        )

        primary_class_scores = score.reshape(
            batch_size,
            1,
            1,
        ).expand(
            batch_size,
            1,
            candidate_count,
        )

        secondary_class_scores = (
            (score * 0.5)
            .reshape(
                batch_size,
                1,
                1,
            )
            .expand(
                batch_size,
                1,
                candidate_count,
            )
        )

        return torch.cat(
            (
                boxes,
                primary_class_scores,
                secondary_class_scores,
            ),
            dim=1,
        )


class FakeDetectionModelWrapper:
    def __init__(
        self,
        model: torch.nn.Module,
        loaded: bool = True,
    ):
        self.model = model
        self.loaded = loaded

    def is_loaded(self) -> bool:
        return self.loaded

    def get_torch_model(self) -> torch.nn.Module:
        return self.model


def _create_test_image(
    size: tuple[int, int] = (8, 8),
    pixel_value: int = 128,
) -> Image.Image:
    return Image.new(
        "RGB",
        size,
        color=(pixel_value, pixel_value, pixel_value),
    )


def _create_parameters(
    epsilon: float,
    alpha: float,
    steps: int,
):
    return SimpleNamespace(
        epsilon=epsilon,
        alpha=alpha,
        steps=steps,
    )


def _create_attack() -> YOLOConfidenceAttack:
    attack = YOLOConfidenceAttack()

    # Reduce el tamaño utilizado durante los tests para que sean rápidos.
    # El comportamiento matemático que queremos validar no cambia.
    attack.attack_size = 16

    return attack


def _capture_adversarial_tensor(
    attack: YOLOConfidenceAttack,
    monkeypatch,
) -> dict[str, torch.Tensor]:
    """
    Captura el tensor adversario antes de convertirlo a imagen PIL.

    Así evitamos que la cuantización RGB de 8 bits afecte a las
    comprobaciones matemáticas.
    """
    captured = {}
    original_tensor_to_image = attack._tensor_to_image

    def capture_tensor(tensor: torch.Tensor) -> Image.Image:
        captured["tensor"] = tensor.detach().clone()
        return original_tensor_to_image(tensor)

    monkeypatch.setattr(
        attack,
        "_tensor_to_image",
        capture_tensor,
    )

    return captured


def _calculate_confidence(
    attack: YOLOConfidenceAttack,
    model: torch.nn.Module,
    tensor: torch.Tensor,
) -> float:
    with torch.no_grad():
        model_input = attack._resize_for_yolo(tensor)
        prediction = model(model_input)
        objective = attack._confidence_objective(prediction)

    return objective.item()


def test_yolo_confidence_returns_expected_result_structure():
    attack = _create_attack()
    model = TinyYOLOModel()
    wrapper = FakeDetectionModelWrapper(model)
    original_image = _create_test_image()

    result = attack.apply(
        model_wrapper=wrapper,
        image=original_image,
        parameters=_create_parameters(
            epsilon=0.1,
            alpha=0.02,
            steps=3,
        ),
    )

    assert set(result) == {
        "adversarial_image",
        "perturbation_image",
        "perturbation_linf",
        "perturbation_l2",
    }

    assert isinstance(result["adversarial_image"], Image.Image)
    assert isinstance(result["perturbation_image"], Image.Image)

    assert result["adversarial_image"].size == original_image.size
    assert result["perturbation_image"].size == original_image.size

    assert result["adversarial_image"].mode == "RGB"
    assert result["perturbation_image"].mode == "RGB"


def test_yolo_confidence_requires_loaded_model():
    attack = _create_attack()

    wrapper = FakeDetectionModelWrapper(
        model=TinyYOLOModel(),
        loaded=False,
    )

    with pytest.raises(ValueError, match="Model is not loaded"):
        attack.apply(
            model_wrapper=wrapper,
            image=_create_test_image(),
            parameters=_create_parameters(
                epsilon=0.1,
                alpha=0.02,
                steps=3,
            ),
        )


def test_yolo_confidence_runs_configured_number_of_steps():
    steps = 4

    attack = _create_attack()
    model = TinyYOLOModel()
    wrapper = FakeDetectionModelWrapper(model)

    attack.apply(
        model_wrapper=wrapper,
        image=_create_test_image(),
        parameters=_create_parameters(
            epsilon=0.1,
            alpha=0.02,
            steps=steps,
        ),
    )

    # El ataque ejecuta una inferencia por cada paso.
    assert model.forward_calls == steps


def test_yolo_confidence_uses_alpha_and_reduces_confidence(
    monkeypatch,
):
    epsilon = 0.2
    alpha = 0.02
    steps = 3

    attack = _create_attack()
    model = TinyYOLOModel()
    wrapper = FakeDetectionModelWrapper(model)
    original_image = _create_test_image(pixel_value=128)

    captured = _capture_adversarial_tensor(
        attack,
        monkeypatch,
    )

    result = attack.apply(
        model_wrapper=wrapper,
        image=original_image,
        parameters=_create_parameters(
            epsilon=epsilon,
            alpha=alpha,
            steps=steps,
        ),
    )

    original_tensor = attack.to_tensor(original_image).unsqueeze(0)
    adversarial_tensor = captured["tensor"]

    expected_perturbation = alpha * steps
    expected_adversarial_tensor = original_tensor - expected_perturbation

    assert torch.allclose(
        adversarial_tensor,
        expected_adversarial_tensor,
        atol=1e-6,
    )

    original_confidence = _calculate_confidence(
        attack,
        model,
        original_tensor,
    )

    adversarial_confidence = _calculate_confidence(
        attack,
        model,
        adversarial_tensor,
    )

    assert adversarial_confidence < original_confidence

    assert result["perturbation_linf"] == pytest.approx(
        expected_perturbation,
        abs=1e-4,
    )


def test_yolo_confidence_projects_perturbation_to_epsilon(
    monkeypatch,
):
    epsilon = 0.08
    alpha = 0.03
    steps = 10

    attack = _create_attack()
    model = TinyYOLOModel()
    wrapper = FakeDetectionModelWrapper(model)
    original_image = _create_test_image(pixel_value=128)

    captured = _capture_adversarial_tensor(
        attack,
        monkeypatch,
    )

    result = attack.apply(
        model_wrapper=wrapper,
        image=original_image,
        parameters=_create_parameters(
            epsilon=epsilon,
            alpha=alpha,
            steps=steps,
        ),
    )

    original_tensor = attack.to_tensor(original_image).unsqueeze(0)
    adversarial_tensor = captured["tensor"]

    perturbation = adversarial_tensor - original_tensor
    observed_linf = torch.max(torch.abs(perturbation)).item()

    expected_adversarial_tensor = original_tensor - epsilon

    assert torch.allclose(
        adversarial_tensor,
        expected_adversarial_tensor,
        atol=1e-6,
    )

    assert observed_linf == pytest.approx(
        epsilon,
        abs=1e-6,
    )

    assert result["perturbation_linf"] == pytest.approx(
        epsilon,
        abs=1e-4,
    )


def test_yolo_confidence_output_remains_in_valid_pixel_range(
    monkeypatch,
):
    attack = _create_attack()
    model = TinyYOLOModel()
    wrapper = FakeDetectionModelWrapper(model)

    original_image = _create_test_image(pixel_value=5)

    captured = _capture_adversarial_tensor(
        attack,
        monkeypatch,
    )

    result = attack.apply(
        model_wrapper=wrapper,
        image=original_image,
        parameters=_create_parameters(
            epsilon=0.2,
            alpha=0.1,
            steps=3,
        ),
    )

    original_tensor = attack.to_tensor(original_image).unsqueeze(0)
    adversarial_tensor = captured["tensor"]

    perturbation = adversarial_tensor - original_tensor
    observed_linf = torch.max(torch.abs(perturbation)).item()

    assert torch.min(adversarial_tensor).item() >= 0.0
    assert torch.max(adversarial_tensor).item() <= 1.0

    assert torch.all(adversarial_tensor == 0.0)

    assert observed_linf <= 0.2 + 1e-7
    assert result["perturbation_linf"] <= 0.2


def test_yolo_confidence_with_zero_epsilon_does_not_modify_image(
    monkeypatch,
):
    attack = _create_attack()
    model = TinyYOLOModel()
    wrapper = FakeDetectionModelWrapper(model)
    original_image = _create_test_image()

    captured = _capture_adversarial_tensor(
        attack,
        monkeypatch,
    )

    result = attack.apply(
        model_wrapper=wrapper,
        image=original_image,
        parameters=_create_parameters(
            epsilon=0.0,
            alpha=0.02,
            steps=3,
        ),
    )

    original_tensor = attack.to_tensor(original_image).unsqueeze(0)
    adversarial_tensor = captured["tensor"]

    assert torch.equal(
        adversarial_tensor,
        original_tensor,
    )

    assert result["adversarial_image"].tobytes() == original_image.tobytes()

    assert result["perturbation_linf"] == 0.0
    assert result["perturbation_l2"] == 0.0


@pytest.mark.parametrize(
    "output_factory",
    [
        lambda tensor: tensor,
        lambda tensor: [tensor],
        lambda tensor: (tensor,),
        lambda tensor: [None, tensor],
        lambda tensor: [[None, tensor]],
        lambda tensor: ([None, tensor],),
    ],
)
def test_extract_prediction_tensor_accepts_supported_structures(
    output_factory,
):
    attack = _create_attack()
    prediction_tensor = torch.ones(1, 6, 10)

    output = output_factory(prediction_tensor)

    extracted = attack._extract_prediction_tensor(output)

    assert extracted is prediction_tensor


def test_extract_prediction_tensor_rejects_output_without_tensor():
    attack = _create_attack()

    with pytest.raises(
        ValueError,
        match="Could not extract YOLO prediction tensor",
    ):
        attack._extract_prediction_tensor(
            [None, "invalid-output"],
        )


def test_extract_prediction_tensor_rejects_yolo26_end_to_end_output():
    attack = _create_attack()
    postprocessed_predictions = torch.zeros(1, 300, 6)
    raw_predictions = {
        "one2many": {"scores": torch.zeros(1, 80, 8400)},
        "one2one": {"scores": torch.zeros(1, 80, 8400)},
    }

    with pytest.raises(
        NotImplementedError,
        match="YOLO26 end-to-end output is not supported",
    ):
        attack._extract_prediction_tensor((postprocessed_predictions, raw_predictions))


def test_confidence_objective_for_standard_yolo_layout():
    attack = _create_attack()

    class_scores = torch.tensor(
        [
            [
                [0.1, 0.8, 0.2, 0.4, 0.7, 0.3, 0.5, 0.2],
                [0.3, 0.2, 0.6, 0.1, 0.4, 0.9, 0.2, 0.8],
            ]
        ],
        dtype=torch.float32,
    )

    boxes = torch.zeros(1, 4, 8)

    prediction_tensor = torch.cat(
        (boxes, class_scores),
        dim=1,
    )

    expected = class_scores.max(dim=1).values.mean()

    result = attack._confidence_objective(prediction_tensor)

    assert result.item() == pytest.approx(
        expected.item(),
        abs=1e-7,
    )


def test_confidence_objective_for_alternative_yolo_layout():
    attack = _create_attack()

    class_scores = torch.tensor(
        [
            [
                [0.1, 0.8, 0.2, 0.4, 0.7, 0.3, 0.5, 0.2],
                [0.3, 0.2, 0.6, 0.1, 0.4, 0.9, 0.2, 0.8],
            ]
        ],
        dtype=torch.float32,
    )

    boxes = torch.zeros(1, 4, 8)

    standard_prediction = torch.cat(
        (boxes, class_scores),
        dim=1,
    )

    alternative_prediction = standard_prediction.transpose(1, 2)

    expected = class_scores.max(dim=1).values.mean()

    result = attack._confidence_objective(
        alternative_prediction,
    )

    assert result.item() == pytest.approx(
        expected.item(),
        abs=1e-7,
    )


def test_confidence_objective_rejects_invalid_dimensions():
    attack = _create_attack()

    invalid_prediction = torch.zeros(1, 10)

    with pytest.raises(
        ValueError,
        match="Unexpected YOLO prediction tensor shape",
    ):
        attack._confidence_objective(invalid_prediction)


@pytest.mark.parametrize(
    "prediction_tensor",
    [
        torch.zeros(1, 4, 10),
        torch.zeros(1, 10, 4),
    ],
)
def test_confidence_objective_rejects_missing_class_scores(
    prediction_tensor,
):
    attack = _create_attack()

    with pytest.raises(
        ValueError,
        match="does not contain class scores",
    ):
        attack._confidence_objective(prediction_tensor)

from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

from ml.attacks.detection_confidence import DetectionConfidenceAttack


class TinyDetectionModel(torch.nn.Module):
    """
    Detector sintético para validar el bucle genérico de supresión de confianza.

    Los cuatro primeros canales representan las cajas.
    Los dos últimos representan puntuaciones de clase.

    La confianza depende directamente de la media de la imagen.
    Por tanto, el descenso aplicado por el ataque debe reducir
    tanto los valores de entrada como la confianza resultante.
    """

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.forward_calls = 0

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.forward_calls += 1

        batch_size = inputs.shape[0]
        candidate_count = 10

        score = inputs.mean(dim=(1, 2, 3)) * self.scale

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
        self.prepare_calls = 0
        self.objective_calls = 0

    def is_loaded(self) -> bool:
        return self.loaded

    def prepare_detection_attack_input(self, image: Image.Image) -> torch.Tensor:
        self.prepare_calls += 1
        return to_tensor(image.convert("RGB")).unsqueeze(0)

    def detection_confidence_objective(
        self,
        prepared_input: torch.Tensor,
    ) -> torch.Tensor:
        self.objective_calls += 1
        prediction_tensor = self.model(prepared_input)

        class_scores = prediction_tensor[:, 4:, :]
        max_class_scores = class_scores.max(dim=1).values

        return max_class_scores.mean()


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


def _create_attack() -> DetectionConfidenceAttack:
    return DetectionConfidenceAttack()


def test_detection_confidence_declares_algorithm_revision():
    assert DetectionConfidenceAttack.revision == "detection-confidence-linf-v2"


def _capture_adversarial_tensor(
    attack: DetectionConfidenceAttack,
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
    wrapper: FakeDetectionModelWrapper,
    tensor: torch.Tensor,
) -> float:
    with torch.no_grad():
        objective = wrapper.detection_confidence_objective(tensor)

    return objective.item()


def test_detection_confidence_returns_expected_result_structure():
    attack = _create_attack()
    model = TinyDetectionModel()
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


def test_detection_confidence_requires_loaded_model():
    attack = _create_attack()

    wrapper = FakeDetectionModelWrapper(
        model=TinyDetectionModel(),
        loaded=False,
    )

    _prepared_image = _create_test_image()
    _prepared_parameters = _create_parameters(
        epsilon=0.1,
        alpha=0.02,
        steps=3,
    )

    with pytest.raises(ValueError, match="Model is not loaded"):
        attack.apply(
            model_wrapper=wrapper,
            image=_prepared_image,
            parameters=_prepared_parameters,
        )


def test_detection_confidence_runs_configured_number_of_steps():
    steps = 4

    attack = _create_attack()
    model = TinyDetectionModel()
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

    # El ataque prepara la imagen una sola vez y delega un objetivo por paso.
    assert wrapper.prepare_calls == 1
    assert wrapper.objective_calls == steps
    assert model.forward_calls == steps


def test_detection_confidence_uses_alpha_and_reduces_confidence(
    monkeypatch,
):
    epsilon = 0.2
    alpha = 0.02
    steps = 3

    attack = _create_attack()
    model = TinyDetectionModel()
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

    original_tensor = to_tensor(original_image).unsqueeze(0)
    adversarial_tensor = captured["tensor"]

    expected_perturbation = alpha * steps
    expected_adversarial_tensor = original_tensor - expected_perturbation

    assert torch.allclose(
        adversarial_tensor,
        expected_adversarial_tensor,
        atol=1e-6,
    )

    original_confidence = _calculate_confidence(
        wrapper,
        original_tensor,
    )

    adversarial_confidence = _calculate_confidence(
        wrapper,
        adversarial_tensor,
    )

    assert adversarial_confidence < original_confidence

    assert result["perturbation_linf"] == pytest.approx(
        expected_perturbation,
        abs=1e-4,
    )


def test_detection_confidence_projects_perturbation_to_epsilon(
    monkeypatch,
):
    epsilon = 0.08
    alpha = 0.03
    steps = 10

    attack = _create_attack()
    model = TinyDetectionModel()
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

    original_tensor = to_tensor(original_image).unsqueeze(0)
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


def test_detection_confidence_output_remains_in_valid_pixel_range(
    monkeypatch,
):
    attack = _create_attack()
    model = TinyDetectionModel()
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

    original_tensor = to_tensor(original_image).unsqueeze(0)
    adversarial_tensor = captured["tensor"]

    perturbation = adversarial_tensor - original_tensor
    observed_linf = torch.max(torch.abs(perturbation)).item()

    assert torch.min(adversarial_tensor).item() >= 0.0
    assert torch.max(adversarial_tensor).item() <= 1.0

    assert torch.all(adversarial_tensor == 0.0)

    assert observed_linf <= 0.2 + 1e-7
    assert result["perturbation_linf"] <= 0.2


def test_detection_confidence_with_zero_epsilon_does_not_modify_image(
    monkeypatch,
):
    attack = _create_attack()
    model = TinyDetectionModel()
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

    original_tensor = to_tensor(original_image).unsqueeze(0)
    adversarial_tensor = captured["tensor"]

    assert torch.equal(
        adversarial_tensor,
        original_tensor,
    )

    assert result["adversarial_image"].tobytes() == original_image.tobytes()

    assert result["perturbation_linf"] == 0.0
    assert result["perturbation_l2"] == 0.0


def test_detection_confidence_does_not_accumulate_model_parameter_gradients():
    attack = _create_attack()
    model = TinyDetectionModel()
    wrapper = FakeDetectionModelWrapper(model)

    attack.apply(
        model_wrapper=wrapper,
        image=_create_test_image(),
        parameters=_create_parameters(
            epsilon=0.1,
            alpha=0.02,
            steps=2,
        ),
    )

    assert model.scale.grad is None


def test_detection_confidence_rejects_non_scalar_model_objective():
    attack = _create_attack()
    wrapper = FakeDetectionModelWrapper(TinyDetectionModel())

    def vector_objective(prepared_input):
        return prepared_input.mean(dim=(1, 2, 3))

    wrapper.detection_confidence_objective = vector_objective

    _prepared_image_2 = _create_test_image()
    _prepared_parameters_2 = _create_parameters(
        epsilon=0.1,
        alpha=0.02,
        steps=1,
    )

    with pytest.raises(
        ValueError,
        match="Detection confidence objective must be a scalar",
    ):
        attack.apply(
            model_wrapper=wrapper,
            image=_prepared_image_2,
            parameters=_prepared_parameters_2,
        )


def test_detection_confidence_rejects_objective_without_input_gradient():
    attack = _create_attack()
    wrapper = FakeDetectionModelWrapper(TinyDetectionModel())

    def detached_objective(prepared_input):
        return torch.tensor(
            1.0,
            device=prepared_input.device,
            requires_grad=True,
        )

    wrapper.detection_confidence_objective = detached_objective

    _prepared_image_3 = _create_test_image()
    _prepared_parameters_3 = _create_parameters(
        epsilon=0.1,
        alpha=0.02,
        steps=1,
    )

    with pytest.raises(
        RuntimeError,
        match="did not produce an input gradient",
    ):
        attack.apply(
            model_wrapper=wrapper,
            image=_prepared_image_3,
            parameters=_prepared_parameters_3,
        )

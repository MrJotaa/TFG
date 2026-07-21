import pytest
import torch
from PIL import Image

import services.attack_service as attack_service
from ml.attacks.base import AdversarialAttackResult
from schemas.attack import AttackRequest, AttackType
from schemas.predict import PredictResponse, TopKPrediction


def _prediction(label: str, confidence: float) -> PredictResponse:
    return PredictResponse(
        prediction=label,
        confidence=confidence,
        top_k=[
            TopKPrediction(
                label=label,
                probability=confidence,
            )
        ],
    )


class FakeClassificationModel:
    id = "fake-classifier"
    task_type = "classification"

    def __init__(self, adversarial_model_input: torch.Tensor):
        self.adversarial_model_input = adversarial_model_input
        self.image_predictions = 0
        self.tensor_predictions = 0

    def predict(self, image: Image.Image) -> PredictResponse:
        self.image_predictions += 1

        if self.image_predictions > 1:
            raise AssertionError("Adversarial PIL must not be used for prediction")

        return _prediction("clean", 0.9)

    def predict_from_prepared_input(
        self,
        prepared_input: torch.Tensor,
    ) -> PredictResponse:
        self.tensor_predictions += 1
        assert prepared_input is self.adversarial_model_input
        return _prediction("adversarial", 0.8)


class FakeAttack:
    id = "fgsm"
    supported_task_types = ["classification"]

    def __init__(self, result):
        self.result = result

    def supports_task_type(self, task_type: str) -> bool:
        return task_type in self.supported_task_types

    def apply(self, model_wrapper, image, parameters, original_class_index=None):
        return self.result


def _structured_result() -> AdversarialAttackResult:
    clean_model_input = torch.zeros(1, 3, 4, 4)
    adversarial_model_input = torch.full_like(clean_model_input, 0.05)

    return AdversarialAttackResult(
        clean_model_input=clean_model_input,
        adversarial_model_input=adversarial_model_input,
        model_space_delta=adversarial_model_input - clean_model_input,
        adversarial_image=Image.new("RGB", (4, 4), color=(13, 13, 13)),
        perturbation_image=Image.new("RGB", (4, 4), color=(255, 255, 255)),
        perturbation_linf=0.05,
        perturbation_l2=0.3464,
    )


def _legacy_result() -> dict:
    return {
        "adversarial_image": Image.new("RGB", (4, 4), color=(13, 13, 13)),
        "perturbation_image": Image.new("RGB", (4, 4), color=(255, 255, 255)),
        "perturbation_linf": 0.05,
        "perturbation_l2": 0.3464,
    }


async def _run_service(
    monkeypatch,
    result,
    model,
    attack_type: AttackType = AttackType.FGSM,
):
    source_image = Image.new("RGB", (8, 6), color=(128, 128, 128))
    saved_images = []

    async def fake_get_image_from_dataset(dataset_id: str, image_id: str):
        return source_image

    def fake_save_result_image(image, result_id, result_path, filename):
        saved_images.append((image, filename))
        return f"/results/{result_id}/{filename}"

    monkeypatch.setattr(
        attack_service,
        "get_image_from_dataset",
        fake_get_image_from_dataset,
    )
    monkeypatch.setattr(
        attack_service,
        "get_active_model_instance",
        lambda: model,
    )
    monkeypatch.setattr(
        attack_service,
        "get_attack",
        lambda attack_type: FakeAttack(result),
    )
    monkeypatch.setattr(
        attack_service,
        "generate_gradcam",
        lambda model, image: {"gradcam_pil_image": image},
    )
    monkeypatch.setattr(
        attack_service,
        "create_result_directory",
        lambda prefix: ("attack-result", "/tmp/attack-result"),
    )
    monkeypatch.setattr(
        attack_service,
        "save_result_image",
        fake_save_result_image,
    )
    monkeypatch.setattr(
        attack_service,
        "save_result_metadata",
        lambda result_path, metadata: None,
    )

    response = await attack_service.apply_adversarial_attack(
        AttackRequest(
            dataset_id="dataset-1",
            image_id="image-1",
            attack_type=attack_type,
        )
    )

    return response, saved_images


@pytest.mark.asyncio
async def test_structured_attack_predicts_from_exact_adversarial_tensor(
    monkeypatch,
):
    result = _structured_result()
    model = FakeClassificationModel(result.adversarial_model_input)
    clean_before = result.clean_model_input.clone()
    adversarial_before = result.adversarial_model_input.clone()
    delta_before = result.model_space_delta.clone()

    response, saved_images = await _run_service(monkeypatch, result, model)

    assert model.image_predictions == 1
    assert model.tensor_predictions == 1
    assert response.adversarial_prediction.prediction == "adversarial"
    assert (result.adversarial_image, "adversarial.png") in saved_images
    assert (result.perturbation_image, "perturbation.png") in saved_images
    assert torch.equal(result.clean_model_input, clean_before)
    assert torch.equal(result.adversarial_model_input, adversarial_before)
    assert torch.equal(result.model_space_delta, delta_before)


@pytest.mark.asyncio
async def test_random_noise_result_keeps_pil_prediction_path(monkeypatch):
    result = _legacy_result()

    class RandomNoiseModel(FakeClassificationModel):
        def predict(self, image: Image.Image) -> PredictResponse:
            self.image_predictions += 1

            if self.image_predictions == 1:
                return _prediction("clean", 0.9)

            assert image is result["adversarial_image"]
            return _prediction("adversarial", 0.8)

        def predict_from_prepared_input(self, prepared_input: torch.Tensor):
            raise AssertionError("Random Noise must keep the PIL prediction path")

    model = RandomNoiseModel(torch.empty(0))

    response, _ = await _run_service(
        monkeypatch,
        result,
        model,
        attack_type=AttackType.RANDOM_NOISE,
    )

    assert model.image_predictions == 2
    assert model.tensor_predictions == 0
    assert response.adversarial_prediction.prediction == "adversarial"

import pytest
import torch
from PIL import Image

import services.attack_service as attack_service
from ml.attacks.base import AdversarialAttackResult
from ml.attacks.presentation import SourceGeometryAttackReconstruction
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
    source_image: Image.Image | None = None,
):
    if source_image is None:
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
    saved_by_filename = {filename: image for image, filename in saved_images}
    assert saved_by_filename["adversarial.png"].size == (8, 6)
    assert saved_by_filename["perturbation.png"].size == (8, 6)
    assert saved_by_filename["adversarial.png"] is not result.adversarial_image
    assert saved_by_filename["perturbation.png"] is not result.perturbation_image
    assert saved_by_filename["adversarial_gradcam.png"] is result.adversarial_image
    assert response.comparison.perturbation_linf == result.perturbation_linf
    assert response.comparison.perturbation_l2 == result.perturbation_l2
    assert torch.equal(result.clean_model_input, clean_before)
    assert torch.equal(result.adversarial_model_input, adversarial_before)
    assert torch.equal(result.model_space_delta, delta_before)


@pytest.mark.parametrize("attack_type", [AttackType.FGSM, AttackType.PGD])
@pytest.mark.parametrize("source_size", [(19, 7), (6, 21), (13, 13)])
@pytest.mark.asyncio
async def test_structured_attack_persists_source_geometry_for_fgsm_and_pgd(
    monkeypatch,
    attack_type,
    source_size,
):
    result = _structured_result()
    model = FakeClassificationModel(result.adversarial_model_input)
    source_image = Image.new("RGB", source_size, color=(128, 128, 128))

    response, saved_images = await _run_service(
        monkeypatch,
        result,
        model,
        attack_type=attack_type,
        source_image=source_image,
    )

    saved_by_filename = {filename: image for image, filename in saved_images}
    assert saved_by_filename["adversarial.png"].size == source_size
    assert saved_by_filename["perturbation.png"].size == source_size
    assert response.adversarial_image_url.endswith("/adversarial.png")
    assert response.perturbation_image_url.endswith("/perturbation.png")


@pytest.mark.asyncio
async def test_structured_attack_projects_exact_delta_once(monkeypatch):
    result = _structured_result()
    model = FakeClassificationModel(result.adversarial_model_input)
    source_image = Image.new("RGB", (9, 5), color=(128, 128, 128))
    reconstructed_adversarial = Image.new("RGB", source_image.size, color=(1, 2, 3))
    reconstructed_perturbation = Image.new("RGB", source_image.size, color=(4, 5, 6))
    projector_calls = 0

    def fake_reconstruct_attack_in_source_geometry(
        source_image: Image.Image,
        model_space_delta: torch.Tensor,
    ) -> SourceGeometryAttackReconstruction:
        nonlocal projector_calls
        projector_calls += 1
        assert source_image is source_image_from_dataset
        assert model_space_delta is result.model_space_delta
        return SourceGeometryAttackReconstruction(
            source_space_delta=torch.empty(0),
            adversarial_image=reconstructed_adversarial,
            perturbation_visualization=reconstructed_perturbation,
        )

    source_image_from_dataset = source_image
    monkeypatch.setattr(
        attack_service,
        "reconstruct_attack_in_source_geometry",
        fake_reconstruct_attack_in_source_geometry,
    )

    _, saved_images = await _run_service(
        monkeypatch,
        result,
        model,
        source_image=source_image,
    )

    saved_by_filename = {filename: image for image, filename in saved_images}
    assert projector_calls == 1
    assert saved_by_filename["adversarial.png"] is reconstructed_adversarial
    assert saved_by_filename["perturbation.png"] is reconstructed_perturbation
    assert model.image_predictions == 1
    assert model.tensor_predictions == 1


@pytest.mark.asyncio
async def test_random_noise_result_keeps_pil_prediction_path(monkeypatch):
    result = _legacy_result()

    def fail_if_reconstruction_is_used(source_image, model_space_delta):
        raise AssertionError("Random Noise must not use source reconstruction")

    monkeypatch.setattr(
        attack_service,
        "reconstruct_attack_in_source_geometry",
        fail_if_reconstruction_is_used,
    )

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

    response, saved_images = await _run_service(
        monkeypatch,
        result,
        model,
        attack_type=AttackType.RANDOM_NOISE,
    )

    assert model.image_predictions == 2
    assert model.tensor_predictions == 0
    assert response.adversarial_prediction.prediction == "adversarial"
    assert (result["adversarial_image"], "adversarial.png") in saved_images
    assert (result["perturbation_image"], "perturbation.png") in saved_images

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image

import services.evaluation_service as evaluation_service
from ml.adapters.art_classification import ARTClassificationAdapter
from ml.attacks.base import AdversarialAttackResult
from ml.attacks.fgsm import FGSMAttack
from ml.attacks.pgd import PGDAttack
from ml.attacks.presentation import perturbation_to_visualization
from ml.attacks.registry import get_attack
from ml.explainability.gradcam import generate_gradcam_from_prepared_input
from ml.models.swin_v2_t.model import SwinV2TinyModel
from schemas.attack import AttackParameters, AttackType
from schemas.datasets import DatasetImage, DatasetResponse
from schemas.evaluation import EvaluationRequest

_CLASS_LABELS = ("red", "green", "blue")
_CLASSIFICATION_ATTACKS = (
    AttackType.FGSM,
    AttackType.RANDOM_NOISE,
    AttackType.PGD,
    AttackType.CARLINI_WAGNER,
    AttackType.DEEPFOOL,
    AttackType.HOP_SKIP_JUMP,
    AttackType.ZOO,
)


class _TinySwinBlock(torch.nn.Module):
    """Lightweight NHWC block that preserves Swin's explainability contract."""

    def __init__(self) -> None:
        super().__init__()
        self.norm1 = torch.nn.LayerNorm(3)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.norm1(inputs)


class _TinySwinLikeClassifier(torch.nn.Module):
    """Cheap differentiable stand-in for Swin used only by contract tests."""

    def __init__(self) -> None:
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Identity(),
            torch.nn.Sequential(_TinySwinBlock()),
        )
        self.norm = torch.nn.LayerNorm(3)
        self.head = torch.nn.Linear(3, len(_CLASS_LABELS), bias=False)

        with torch.no_grad():
            self.head.weight.copy_(torch.eye(len(_CLASS_LABELS)))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        channels_last = inputs.permute(0, 2, 3, 1)
        features = self.features[-1][-1](channels_last)
        features = self.norm(features)
        pooled = features.mean(dim=(1, 2))
        return self.head(pooled)


class _StructuredClassificationAttack:
    id = "swin-contract-attack"
    supported_task_types = ("classification",)

    def supports_task_type(self, task_type: str) -> bool:
        return task_type in self.supported_task_types

    def apply(
        self,
        model_wrapper,
        image: Image.Image,
        parameters,
        original_class_index: int | None = None,
    ) -> AdversarialAttackResult:
        clean_model_input = model_wrapper.prepare_attack_input(image).detach()
        model_space_delta = torch.full_like(clean_model_input, 0.01)
        adversarial_model_input = torch.clamp(
            clean_model_input + model_space_delta,
            min=0.0,
            max=1.0,
        )
        model_space_delta = adversarial_model_input - clean_model_input

        adversarial_image = to_pil_image(adversarial_model_input[0].cpu())

        return AdversarialAttackResult(
            clean_model_input=clean_model_input,
            adversarial_model_input=adversarial_model_input,
            model_space_delta=model_space_delta,
            adversarial_image=adversarial_image,
            perturbation_image=perturbation_to_visualization(model_space_delta),
            perturbation_linf=round(model_space_delta.abs().max().item(), 4),
            perturbation_l2=round(
                torch.linalg.vector_norm(model_space_delta).item(),
                4,
            ),
        )


def _loaded_swin() -> SwinV2TinyModel:
    model = SwinV2TinyModel()
    model.device = torch.device("cpu")
    model.model = _TinySwinLikeClassifier().eval()
    model.categories = _CLASS_LABELS
    return model


def _source_image() -> Image.Image:
    return Image.new("RGB", (37, 19), color=(180, 90, 30))


def test_swin_prediction_contract_uses_transformer_geometry_without_real_weights():
    model = _loaded_swin()
    source_image = _source_image()

    prepared_input = model.prepare_attack_input(source_image)
    detailed = model.predict_detailed_from_prepared_input(prepared_input, top_k=5)
    public = model.predict_from_prepared_input(prepared_input)

    assert prepared_input.shape == (1, 3, 256, 256)
    assert prepared_input.dtype == torch.float32
    assert prepared_input.min().item() >= 0.0
    assert prepared_input.max().item() <= 1.0

    assert detailed.class_label in _CLASS_LABELS
    assert detailed.class_index in range(len(_CLASS_LABELS))
    assert len(detailed.probabilities) == len(_CLASS_LABELS)
    assert len(detailed.top_k) == len(_CLASS_LABELS)
    assert sum(detailed.probabilities) == pytest.approx(1.0)
    assert public.prediction == detailed.class_label
    assert len(public.top_k) == len(_CLASS_LABELS)


def test_swin_gradcam_contract_projects_nhwc_activations_to_source_geometry():
    model = _loaded_swin()
    source_image = _source_image()
    prepared_input = model.prepare_attack_input(source_image)

    result = generate_gradcam_from_prepared_input(
        model_wrapper=model,
        model_input=prepared_input,
        visualization_image=source_image,
    )

    assert result["prediction"] in _CLASS_LABELS
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["gradcam_pil_image"].mode == "RGB"
    assert result["gradcam_pil_image"].size == source_image.size
    assert len(model.get_gradcam_target_layer()._forward_hooks) == 0


@pytest.mark.parametrize(
    ("attack", "parameters", "epsilon"),
    [
        (FGSMAttack(), AttackParameters(epsilon=0.01), 0.01),
        (
            PGDAttack(),
            AttackParameters(epsilon=0.02, alpha=0.01, steps=2),
            0.02,
        ),
    ],
)
def test_swin_supports_native_gradient_attacks_in_pixel_space(
    attack,
    parameters,
    epsilon,
):
    model = _loaded_swin()
    source_image = _source_image()
    original = model.predict_detailed(source_image)

    result = attack.apply(
        model_wrapper=model,
        image=source_image,
        parameters=parameters,
        original_class_index=original.class_index,
    )

    assert result.clean_model_input.shape == (1, 3, 256, 256)
    assert result.adversarial_model_input.shape == result.clean_model_input.shape
    assert result.model_space_delta.shape == result.clean_model_input.shape
    assert result.adversarial_model_input.min().item() >= 0.0
    assert result.adversarial_model_input.max().item() <= 1.0
    assert torch.equal(
        result.model_space_delta,
        result.adversarial_model_input - result.clean_model_input,
    )
    assert result.model_space_delta.abs().max().item() <= epsilon + 1e-6
    assert result.adversarial_image.size == (256, 256)


def test_swin_art_adapter_exposes_predictions_and_gradients_for_shared_attacks():
    model = _loaded_swin()
    prepared_input = model.prepare_attack_input(_source_image())
    adapter = ARTClassificationAdapter(model)
    art_input = adapter.to_art_input(prepared_input)
    label = model.predict_detailed_from_prepared_input(prepared_input).class_index
    labels = np.eye(len(_CLASS_LABELS), dtype=np.float32)[[label]]

    predictions = adapter.estimator.predict(art_input)
    gradients = adapter.estimator.loss_gradient(art_input, labels)

    assert adapter.input_shape == (3, 256, 256)
    assert adapter.nb_classes == len(_CLASS_LABELS)
    assert predictions.shape == (1, len(_CLASS_LABELS))
    assert gradients.shape == art_input.shape
    assert np.isfinite(predictions).all()
    assert np.isfinite(gradients).all()
    assert not np.allclose(gradients, 0.0)


@pytest.mark.parametrize("attack_type", _CLASSIFICATION_ATTACKS)
def test_every_registered_classification_attack_accepts_swin_task_type(attack_type):
    model = _loaded_swin()
    attack = get_attack(attack_type)

    assert attack.supports_task_type(model.task_type)


@pytest.mark.asyncio
async def test_swin_runs_through_classification_evaluation_without_model_specific_path(
    monkeypatch,
):
    model = _loaded_swin()
    source_image = _source_image()
    attack = _StructuredClassificationAttack()
    dataset = DatasetResponse(
        id="dataset-swin",
        name="Swin contract dataset",
        owner_id="user-1",
        image_count=1,
        images=[
            DatasetImage(
                id="image-1",
                filename="image-1.png",
                label=None,
            )
        ],
    )
    saved_inputs = {}
    saved_images = {}

    async def fake_get_dataset(dataset_id: str):
        assert dataset_id == dataset.id
        return dataset

    async def fake_get_image_from_dataset(dataset_id: str, image_id: str):
        assert dataset_id == dataset.id
        assert image_id == "image-1"
        return source_image.copy()

    def fake_save_result_image(image, result_id, result_path, filename):
        saved_images[filename] = image.copy()
        return f"/results/{result_id}/{filename}"

    def fake_save_evaluation_model_inputs(**kwargs):
        saved_inputs.update(kwargs)

    monkeypatch.setattr(evaluation_service, "get_dataset", fake_get_dataset)
    monkeypatch.setattr(
        evaluation_service,
        "get_image_from_dataset",
        fake_get_image_from_dataset,
    )
    monkeypatch.setattr(
        evaluation_service,
        "get_active_model_instance",
        lambda: model,
    )
    monkeypatch.setattr(evaluation_service, "get_attack", lambda attack_type: attack)
    monkeypatch.setattr(
        evaluation_service,
        "resolve_attack_parameters",
        lambda attack, parameters: parameters,
    )
    monkeypatch.setattr(
        evaluation_service,
        "create_attack_runner",
        lambda **kwargs: attack,
    )
    monkeypatch.setattr(
        evaluation_service,
        "create_result_directory",
        lambda prefix: ("evaluation-swin", "/tmp/evaluation-swin"),
    )
    monkeypatch.setattr(
        evaluation_service,
        "save_result_image",
        fake_save_result_image,
    )
    monkeypatch.setattr(
        evaluation_service,
        "save_result_metadata",
        lambda **kwargs: "/tmp/evaluation-swin/metadata.json",
    )
    monkeypatch.setattr(
        evaluation_service,
        "save_evaluation_model_inputs",
        fake_save_evaluation_model_inputs,
    )

    response = await evaluation_service.evaluate_dataset(
        EvaluationRequest(
            dataset_id=dataset.id,
            attack_type=AttackType.FGSM,
            parameters=AttackParameters(epsilon=0.01),
        )
    )

    assert response.model_id == "swin_v2_t"
    assert response.model_name == model.name
    assert response.summary.total_images == 1
    assert len(response.results) == 1
    assert saved_inputs["model_id"] == "swin_v2_t"
    assert saved_inputs["clean_model_input"].shape == (1, 3, 256, 256)
    assert saved_inputs["adversarial_model_input"].shape == (1, 3, 256, 256)
    assert saved_images["image-1_adversarial.png"].size == source_image.size
    assert saved_images["image-1_perturbation.png"].size == source_image.size

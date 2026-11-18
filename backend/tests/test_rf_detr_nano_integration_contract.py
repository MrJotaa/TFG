import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import to_tensor

import services.detection_attack_service as detection_attack_service
import services.detection_evaluation_service as detection_evaluation_service
from ml.attacks.detection_confidence import DetectionConfidenceAttack
from ml.attacks.detection_localization_pgd import DetectionLocalizationPGDAttack
from ml.attacks.registry import get_attack
from ml.explainability.gradcam import generate_detection_eigencam
from ml.models.definitions import build_model_registry
from ml.models.detection import DetectionBox, DetectionResult
from ml.models.detection_localization import DetectionLocalizationReferences
from ml.models.detection_inference import DetectionEvaluationInferenceConfiguration
from ml.models.rf_detr.catalog import build_rf_detr_class_catalog
from ml.models.rf_detr.config import RfDetrDetectionModelConfig
from ml.models.rf_detr.model import RfDetrDetectionModel
from schemas.attack import AttackParameters, AttackType
from schemas.detection_attack import DetectionAttackRequest
from schemas.detection_evaluation import DetectionEvaluationRequest
from services.annotation_importers.base import (
    CanonicalCategory,
    CanonicalObjectAnnotation,
)
from services.detection_annotation_snapshot import (
    DetectionSnapshotImage,
    MaterializedDetectionAnnotationSnapshot,
)

RF_DETR_MODEL_ID = "rf_detr_nano"
REAL_SMOKE_ENABLED = os.getenv("RUN_RF_DETR_REAL_SMOKE") == "1"
_DETECTION_ATTACKS = (
    AttackType.RANDOM_NOISE,
    AttackType.YOLO_CONFIDENCE,
    AttackType.DETECTION_LOCALIZATION_PGD,
)


class _ContractSpatialProjection(torch.nn.Module):
    """Cheap RF-DETR-like spatial projection for EigenCAM contract tests."""

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        pooled = F.adaptive_avg_pool2d(pixel_values, output_size=(24, 24))
        mean_channel = pooled.mean(dim=1, keepdim=True)
        channel_deltas = torch.stack(
            (
                pooled[:, 0] - pooled[:, 1],
                pooled[:, 1] - pooled[:, 2],
                pooled[:, 2] - pooled[:, 0],
            ),
            dim=1,
        )
        return torch.cat((pooled, mean_channel, channel_deltas), dim=1)


class _ContractProjector(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer_norm = _ContractSpatialProjection()


class _ContractBackbone(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projector = _ContractProjector()


class _ContractCore(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = _ContractBackbone()


class _TinyRfDetrLikeProvider(torch.nn.Module):
    """Cheap spatial detector that preserves RF-DETR's application contracts."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.model = _ContractCore()
        self.config = SimpleNamespace(
            id2label={0: "N/A", 1: "person", 2: "N/A", 3: "car"}
        )

    def forward(
        self,
        pixel_values: torch.Tensor | None = None,
        pixel_mask: torch.Tensor | None = None,
        **_kwargs,
    ):
        if pixel_values is None:
            raise ValueError("pixel_values are required")

        spatial_features = self.model.backbone.projector.layer_norm(pixel_values)
        signal = spatial_features.mean(dim=(1, 2, 3)) * self.scale

        low = torch.full_like(signal, -4.0)
        query_person = torch.stack(
            (low, signal, low, signal * 0.5),
            dim=-1,
        )
        query_car = torch.stack(
            (low, signal * 0.6, low, signal * 0.8),
            dim=-1,
        )
        logits = torch.stack((query_person, query_car), dim=1)

        return SimpleNamespace(logits=logits)


class _TinyRfDetrProcessor:
    do_resize = True
    do_rescale = True
    do_normalize = True
    size = {"height": 384, "width": 384}
    image_mean = [0.485, 0.456, 0.406]
    image_std = [0.229, 0.224, 0.225]

    def __call__(self, *, images: Image.Image, return_tensors: str):
        assert return_tensors == "pt"
        source = to_tensor(images.convert("RGB")).unsqueeze(0)
        resized = F.interpolate(
            source,
            size=(384, 384),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        mean = torch.tensor(self.image_mean, dtype=resized.dtype).view(1, 3, 1, 1)
        std = torch.tensor(self.image_std, dtype=resized.dtype).view(1, 3, 1, 1)
        return {
            "pixel_values": (resized - mean) / std,
            "pixel_mask": torch.ones((1, 384, 384), dtype=torch.long),
        }

    def post_process_object_detection(
        self,
        outputs,
        *,
        threshold: float,
        target_sizes: torch.Tensor,
        top_k: int,
    ):
        height, width = target_sizes[0].detach().cpu().tolist()
        candidate_scores = torch.stack(
            (
                outputs.logits[0, 0, 1].sigmoid(),
                outputs.logits[0, 1, 3].sigmoid(),
            )
        )
        candidate_labels = torch.tensor(
            [1, 3],
            device=candidate_scores.device,
            dtype=torch.long,
        )
        candidate_boxes = torch.tensor(
            [
                [0.10 * width, 0.10 * height, 0.60 * width, 0.80 * height],
                [0.55 * width, 0.20 * height, 0.90 * width, 0.75 * height],
            ],
            device=candidate_scores.device,
            dtype=candidate_scores.dtype,
        )

        keep = candidate_scores >= threshold
        scores = candidate_scores[keep][:top_k]
        labels = candidate_labels[keep][:top_k]
        boxes = candidate_boxes[keep][:top_k]
        return [{"scores": scores, "labels": labels, "boxes": boxes}]


def _loaded_rf_detr() -> RfDetrDetectionModel:
    model = RfDetrDetectionModel(
        RfDetrDetectionModelConfig(
            id=RF_DETR_MODEL_ID,
            name="RF-DETR Nano contract",
            revision="rf-detr-contract-v1",
            checkpoint="contract/rf-detr-nano",
            checkpoint_revision="contract-revision",
            confidence_threshold=0.25,
        )
    )
    model.device = torch.device("cpu")
    model.model = _TinyRfDetrLikeProvider().eval()
    model.image_processor = _TinyRfDetrProcessor()
    model._id2label = dict(model.model.config.id2label)
    model._class_catalog = build_rf_detr_class_catalog(
        model_id=model.id,
        model_name=model.name,
        id2label=model._id2label,
    )
    return model


def _source_image() -> Image.Image:
    width, height = 32, 24
    x = np.linspace(24, 232, width, dtype=np.float32)[None, :]
    y = np.linspace(32, 224, height, dtype=np.float32)[:, None]
    red = np.broadcast_to(x, (height, width))
    green = np.broadcast_to(y, (height, width))
    blue = (0.65 * red + 0.35 * green) % 255
    rgb = np.stack((red, green, blue), axis=-1).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def _supervised_snapshot() -> MaterializedDetectionAnnotationSnapshot:
    return MaterializedDetectionAnnotationSnapshot(
        dataset_id="dataset-rf-detr-contract",
        dataset_revision=4,
        snapshot_id="snapshot-rf-detr-contract",
        images=(
            DetectionSnapshotImage(
                image_id="image-1",
                filename="image-1.png",
                width=32,
                height=24,
                review_status="reviewed_with_objects",
            ),
        ),
        categories=(
            CanonicalCategory(
                internal_id="category-person",
                external_id=91,
                name="person",
                supercategory=None,
            ),
        ),
        annotations=(
            CanonicalObjectAnnotation(
                external_id=1001,
                image_id="image-1",
                category_internal_id="category-person",
                bbox_x1=3.2,
                bbox_y1=2.4,
                bbox_x2=19.2,
                bbox_y2=19.2,
                area=268.8,
                iscrowd=0,
                object_internal_id="object-rf-detr-1",
            ),
        ),
    )


def test_rf_detr_detection_contract_preserves_source_geometry_and_visualization():
    model = _loaded_rf_detr()
    source_image = _source_image()

    result, annotated = model.detect_with_visualization(source_image)

    assert isinstance(result, DetectionResult)
    assert result.source_width == source_image.width
    assert result.source_height == source_image.height
    assert [(item.class_id, item.class_name) for item in result.detections] == [
        (1, "person"),
        (3, "car"),
    ]
    assert all(0.0 <= item.confidence <= 1.0 for item in result.detections)
    assert annotated.size == source_image.size
    assert annotated.mode == "RGB"
    assert annotated.tobytes() != source_image.tobytes()


@pytest.mark.parametrize("attack_type", _DETECTION_ATTACKS)
def test_every_registered_detection_attack_accepts_rf_detr_task_type(attack_type):
    model = _loaded_rf_detr()
    attack = get_attack(attack_type)

    assert attack.supports_task_type(model.task_type)


def test_rf_detr_detection_confidence_attack_reduces_objective_in_source_space():
    model = _loaded_rf_detr()
    source_image = _source_image()
    attack = DetectionConfidenceAttack()

    clean_input = model.prepare_detection_attack_input(source_image)
    clean_objective = model.detection_confidence_objective(clean_input).item()

    result = attack.apply(
        model_wrapper=model,
        image=source_image,
        parameters=AttackParameters(epsilon=0.03, alpha=0.01, steps=2),
    )

    adversarial_input = model.prepare_detection_attack_input(
        result["adversarial_image"]
    )
    adversarial_objective = model.detection_confidence_objective(
        adversarial_input
    ).item()

    assert adversarial_objective < clean_objective
    assert result["perturbation_linf"] <= 0.03
    assert result["adversarial_image"].size == source_image.size
    assert result["perturbation_image"].size == source_image.size


def test_rf_detr_eigencam_uses_spatial_contract_and_releases_hooks():
    model = _loaded_rf_detr()
    source_image = _source_image()
    target_layer = model.get_explainability_target_layers()[0]

    explanation = generate_detection_eigencam(model, source_image)

    assert explanation.size == source_image.size
    assert explanation.mode == "RGB"
    assert explanation.tobytes() != source_image.tobytes()
    assert len(target_layer._forward_hooks) == 0


@pytest.mark.asyncio
async def test_rf_detr_runs_through_detection_attack_service_with_eigencam(
    monkeypatch,
):
    model = _loaded_rf_detr()
    source_image = _source_image()
    saved_images = {}
    saved_metadata = {}

    async def fake_get_image_from_dataset(
        dataset_id: str,
        image_id: str,
        owner_id: str | None = None,
    ):
        assert owner_id is None
        assert dataset_id == "dataset-rf-detr-contract"
        assert image_id == "image-1"
        return source_image.copy()

    def fake_save_result_image(*, image, result_id, result_path, filename):
        assert result_id == "detection_attack_rf_detr_contract"
        assert result_path == "/tmp/detection_attack_rf_detr_contract"
        saved_images[filename] = image.copy()
        return f"/results/{result_id}/{filename}"

    monkeypatch.setattr(
        detection_attack_service,
        "get_image_from_dataset",
        fake_get_image_from_dataset,
    )
    monkeypatch.setattr(
        detection_attack_service,
        "get_active_model_instance",
        lambda: model,
    )
    monkeypatch.setattr(
        detection_attack_service,
        "create_result_directory",
        lambda prefix: (
            "detection_attack_rf_detr_contract",
            "/tmp/detection_attack_rf_detr_contract",
        ),
    )
    monkeypatch.setattr(
        detection_attack_service,
        "save_result_image",
        fake_save_result_image,
    )
    monkeypatch.setattr(
        detection_attack_service,
        "save_result_metadata",
        lambda *, result_path, metadata: saved_metadata.update(metadata),
    )

    response = await detection_attack_service.apply_detection_attack(
        DetectionAttackRequest(
            dataset_id="dataset-rf-detr-contract",
            image_id="image-1",
            attack_type=AttackType.YOLO_CONFIDENCE,
            parameters=AttackParameters(epsilon=0.02, alpha=0.01, steps=1),
        )
    )

    assert response.result_id == "detection_attack_rf_detr_contract"
    assert response.attack_type == AttackType.YOLO_CONFIDENCE
    assert response.original_explanation_image_url is not None
    assert response.adversarial_explanation_image_url is not None
    assert response.comparison.perturbation_linf <= 0.02
    assert len(response.original_detections) == 2
    assert len(response.adversarial_detections) == 2
    assert set(saved_images) == {
        "original_annotated.png",
        "adversarial.png",
        "adversarial_annotated.png",
        "perturbation.png",
        "original_explanation.png",
        "adversarial_explanation.png",
    }
    assert all(image.size == source_image.size for image in saved_images.values())
    assert saved_metadata["model_id"] == RF_DETR_MODEL_ID
    assert saved_metadata["explanation_type"] == "eigencam"
    assert len(model.get_explainability_target_layers()[0]._forward_hooks) == 0


@pytest.mark.asyncio
async def test_rf_detr_runs_through_supervised_detection_evaluation(
    monkeypatch,
):
    model = _loaded_rf_detr()
    source_image = _source_image()
    snapshot = _supervised_snapshot()
    persisted = []
    saved_adversarial_images = []
    dataset = SimpleNamespace(
        owner_id="user-1",
        images=[SimpleNamespace(id="image-1", filename="image-1.png")],
    )

    async def fake_get_dataset(dataset_id: str):
        assert dataset_id == "dataset-rf-detr-contract"
        return dataset

    async def fake_get_image_from_dataset(dataset_id: str, image_id: str):
        assert dataset_id == "dataset-rf-detr-contract"
        assert image_id == "image-1"
        return source_image.copy()

    async def fake_read_snapshot(dataset_id: str):
        assert dataset_id == "dataset-rf-detr-contract"
        return snapshot

    async def fake_persist(*, response, request, owner_id):
        persisted.append((response, request, owner_id))

    monkeypatch.setattr(
        detection_evaluation_service,
        "get_dataset",
        fake_get_dataset,
    )
    monkeypatch.setattr(
        detection_evaluation_service,
        "get_image_from_dataset",
        fake_get_image_from_dataset,
    )
    monkeypatch.setattr(
        detection_evaluation_service,
        "read_detection_annotation_snapshot",
        fake_read_snapshot,
    )
    monkeypatch.setattr(
        detection_evaluation_service,
        "get_active_model_instance",
        lambda: model,
    )
    monkeypatch.setattr(
        detection_evaluation_service,
        "new_detection_evaluation_id",
        lambda: "detection_evaluation_rf_detr_contract",
    )
    monkeypatch.setattr(
        detection_evaluation_service,
        "save_detection_evaluation_adversarial_image",
        lambda **kwargs: (
            saved_adversarial_images.append(kwargs["image"].copy())
            or (
                f"/results/{kwargs['evaluation_id']}/"
                f"{kwargs['image_id']}_adversarial.png"
            )
        ),
    )
    monkeypatch.setattr(
        detection_evaluation_service,
        "persist_detection_evaluation",
        fake_persist,
    )
    monkeypatch.setattr(
        detection_evaluation_service,
        "remove_detection_evaluation_visual_artifacts",
        lambda result_id: None,
    )

    response = await detection_evaluation_service.evaluate_detection_dataset(
        DetectionEvaluationRequest(
            dataset_id="dataset-rf-detr-contract",
            attack_type=AttackType.RANDOM_NOISE,
            parameters=AttackParameters(epsilon=0.02, seed=7),
        )
    )

    assert response.result_id == "detection_evaluation_rf_detr_contract"
    assert response.model_id == RF_DETR_MODEL_ID
    assert response.dataset_revision == 4
    assert response.annotation_snapshot_id == "snapshot-rf-detr-contract"
    assert response.summary.total_images == 1
    assert len(response.results) == 1
    assert response.results[0].adversarial_image_url is not None
    assert response.supervised.available is True
    assert response.supervised.original.aggregate.map_50_95 == pytest.approx(1.0)
    assert (
        response.supervised.original.per_class[0].reference.model_class_name == "person"
    )
    assert len(saved_adversarial_images) == 1
    assert saved_adversarial_images[0].size == source_image.size
    assert len(persisted) == 1
    assert persisted[0][2] == "user-1"


@pytest.mark.real_model
@pytest.mark.skipif(
    not REAL_SMOKE_ENABLED,
    reason=(
        "Set RUN_RF_DETR_REAL_SMOKE=1 to execute the real RF-DETR Nano "
        "checkpoint smoke test"
    ),
)
def test_real_rf_detr_nano_checkpoint_supports_application_contracts():
    model = build_model_registry().get_model(RF_DETR_MODEL_ID)
    assert isinstance(model, RfDetrDetectionModel)
    source_image = _source_image().resize((96, 64), Image.Resampling.BILINEAR)

    try:
        model.load()

        catalog = model.get_detection_class_catalog()
        assert catalog.model_id == RF_DETR_MODEL_ID
        assert len(catalog.classes) == 80

        detection_result, annotated = model.detect_with_visualization(source_image)
        assert isinstance(detection_result, DetectionResult)
        assert detection_result.source_width == source_image.width
        assert detection_result.source_height == source_image.height
        assert annotated.size == source_image.size

        evaluation_result = model.detect_for_evaluation(
            source_image,
            DetectionEvaluationInferenceConfiguration(),
        )
        assert isinstance(evaluation_result, DetectionResult)
        assert evaluation_result.source_width == source_image.width
        assert evaluation_result.source_height == source_image.height

        prepared_input = model.prepare_detection_attack_input(source_image)
        prepared_input.requires_grad_(True)
        objective = model.detection_confidence_objective(prepared_input)
        (input_gradient,) = torch.autograd.grad(objective, prepared_input)

        assert objective.ndim == 0
        assert torch.isfinite(objective)
        assert input_gradient.shape == prepared_input.shape
        assert torch.isfinite(input_gradient).all()
        assert torch.count_nonzero(input_gradient) > 0
        assert all(
            parameter.grad is None for parameter in model.get_torch_model().parameters()
        )

        # A3: comprobar en el checkpoint fijado las pérdidas oficiales
        # y el gradiente de localización en geometría de origen.
        # Una referencia COCO real sería preferible; usamos una caja válida
        # conocida para verificar el contrato sin depender de detecciones.
        localization_references = (
            DetectionLocalizationReferences.from_supervised_objects(
                source_width=source_image.width,
                source_height=source_image.height,
                objects=((1, DetectionBox(10.0, 10.0, 60.0, 45.0)),),
            )
        )
        localization_input = model.prepare_detection_attack_input(
            source_image
        ).requires_grad_(True)
        localization_loss = model.detection_localization_objective(
            localization_input, localization_references
        )
        (localization_gradient,) = torch.autograd.grad(
            localization_loss, localization_input
        )
        assert localization_loss.ndim == 0
        assert torch.isfinite(localization_loss)
        assert localization_gradient.shape == localization_input.shape
        assert torch.isfinite(localization_gradient).all()
        assert torch.count_nonzero(localization_gradient) > 0
        assert all(
            parameter.grad is None
            for parameter in model.get_torch_model().parameters()
        )

        explanation = generate_detection_eigencam(model, source_image)
        assert explanation.size == source_image.size
        assert explanation.mode == "RGB"
        assert len(model.get_explainability_target_layers()[0]._forward_hooks) == 0

        attack_result = DetectionConfidenceAttack().apply(
            model_wrapper=model,
            image=source_image,
            parameters=AttackParameters(epsilon=0.01, alpha=0.01, steps=1),
        )
        assert attack_result["perturbation_linf"] <= 0.01
        assert attack_result["adversarial_image"].size == source_image.size

        # A4: smoke completo del ataque real (una iteración). Si el detector
        # no encuentra objetos sobre esta imagen artificial, el comportamiento
        # científicamente correcto es dejarla intacta.
        localization_attack_result = DetectionLocalizationPGDAttack().apply(
            model_wrapper=model,
            image=source_image,
            parameters=AttackParameters(epsilon=0.03, alpha=0.01, steps=1),
        )
        assert localization_attack_result["adversarial_image"].size == source_image.size
        assert localization_attack_result["perturbation_image"].size == source_image.size
        assert 0 <= localization_attack_result["perturbation_linf"] <= 0.03
        assert all(
            parameter.grad is None
            for parameter in model.get_torch_model().parameters()
        )
    finally:
        model.unload()

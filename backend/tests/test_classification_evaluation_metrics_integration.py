from collections.abc import Sequence
from weakref import ref

import pytest
import torch
from PIL import Image

import services.evaluation_service as evaluation_service
from ml.attacks.base import (
    AdversarialAttackResult,
    AdversarialEvaluationResult,
)
from ml.models.classification import (
    ClassificationClassScore,
    DetailedClassificationPrediction,
)
from schemas.attack import AttackType
from schemas.datasets import DatasetImage, DatasetResponse
from schemas.evaluation import EvaluationRequest

CLASS_LABELS = (
    "cat",
    "dog",
    "car",
)


class FakeClassificationModel:
    id = "test-classifier"
    name = "Test classifier"
    task_type = "classification"
    evaluation_attack_micro_batch_size = 4

    def __init__(
        self,
        predictions: Sequence[DetailedClassificationPrediction],
    ):
        self._predictions = iter(predictions)
        self.prepared_inputs: list[torch.Tensor] = []
        self.prepared_image_inputs: list[torch.Tensor] = []

    def prepare_attack_input(
        self,
        image: Image.Image,
    ) -> torch.Tensor:
        assert image.mode == "RGB"
        prepared_input = torch.zeros((1, 3, 224, 224))
        self.prepared_image_inputs.append(prepared_input)
        return prepared_input

    def predict_detailed_from_prepared_input(
        self,
        prepared_input: torch.Tensor,
        top_k: int = 5,
    ) -> DetailedClassificationPrediction:
        assert top_k == 5
        self.prepared_inputs.append(prepared_input)
        return next(self._predictions)

    def get_class_labels(self) -> tuple[str, ...]:
        return CLASS_LABELS

    def get_torch_model(self):
        raise NotImplementedError


class FakeClassificationAttack:
    id = "test-attack"
    supported_task_types = ["classification"]

    def __init__(
        self,
        perturbations: Sequence[tuple[float, float]],
    ):
        self._perturbations = iter(perturbations)
        self.original_class_indices: list[int | None] = []
        self.adversarial_model_inputs: list[torch.Tensor] = []
        self.model_space_deltas: list[torch.Tensor] = []
        self.temporary_adversarial_images: list[Image.Image] = []
        self.temporary_perturbation_images: list[Image.Image] = []

    def supports_task_type(self, task_type: str) -> bool:
        return task_type in self.supported_task_types

    def apply(
        self,
        model_wrapper,
        image: Image.Image,
        parameters,
        original_class_index: int | None = None,
    ):
        assert model_wrapper.task_type == "classification"
        assert parameters.epsilon == pytest.approx(0.03)

        self.original_class_indices.append(original_class_index)
        perturbation_linf, perturbation_l2 = next(self._perturbations)

        clean_model_input = torch.zeros(1, 3, 224, 224)
        adversarial_model_input = clean_model_input.clone()
        model_space_delta = adversarial_model_input - clean_model_input
        self.adversarial_model_inputs.append(adversarial_model_input)
        self.model_space_deltas.append(model_space_delta)

        adversarial_image = Image.new(
            "RGB",
            (224, 224),
            color=(20, 30, 40),
        )
        perturbation_image = Image.new(
            "RGB",
            (224, 224),
            color=(127, 127, 127),
        )
        self.temporary_adversarial_images.append(adversarial_image)
        self.temporary_perturbation_images.append(perturbation_image)

        return AdversarialAttackResult(
            clean_model_input=clean_model_input,
            adversarial_model_input=adversarial_model_input,
            model_space_delta=model_space_delta,
            adversarial_image=adversarial_image,
            perturbation_image=perturbation_image,
            perturbation_linf=perturbation_linf,
            perturbation_l2=perturbation_l2,
        )


class FastPathClassificationAttack(FakeClassificationAttack):
    def __init__(
        self,
        perturbations: Sequence[tuple[float, float]],
    ):
        super().__init__(perturbations)
        self.evaluation_prepared_inputs: list[torch.Tensor] = []

    def apply(
        self,
        model_wrapper,
        image: Image.Image,
        parameters,
        original_class_index: int | None = None,
    ):
        raise AssertionError("Evaluation should use apply_for_evaluation")

    def apply_for_evaluation(
        self,
        model_wrapper,
        image: Image.Image,
        prepared_input: torch.Tensor,
        parameters,
        original_class_index: int | None = None,
    ):
        self.evaluation_prepared_inputs.append(prepared_input)
        return FakeClassificationAttack.apply(
            self,
            model_wrapper=model_wrapper,
            image=image,
            parameters=parameters,
            original_class_index=original_class_index,
        )


class BatchFastPathClassificationAttack:
    id = "fgsm"
    supported_task_types = ["classification"]
    evaluation_micro_batch_size = 2

    def __init__(self):
        self.batch_sizes: list[int] = []
        self.original_class_index_batches: list[tuple[int | None, ...]] = []

    def supports_task_type(self, task_type: str) -> bool:
        return task_type in self.supported_task_types

    def apply(self, **kwargs):
        raise AssertionError("Evaluation should use the micro-batch fast path")

    def apply_for_evaluation(self, **kwargs):
        raise AssertionError("Evaluation should use apply_many_for_evaluation")

    def apply_many_for_evaluation(
        self,
        model_wrapper,
        images,
        prepared_inputs,
        parameters,
        original_class_indices,
    ):
        assert model_wrapper.task_type == "classification"
        assert all(
            sample_parameters.epsilon == pytest.approx(0.03)
            for sample_parameters in parameters
        )
        self.batch_sizes.append(len(images))
        self.original_class_index_batches.append(tuple(original_class_indices))

        results = []
        for prepared_input in prepared_inputs:
            adversarial_model_input = prepared_input.detach().clone()
            model_space_delta = torch.zeros_like(adversarial_model_input)
            results.append(
                AdversarialEvaluationResult(
                    clean_model_input=prepared_input.detach().clone(),
                    adversarial_model_input=adversarial_model_input,
                    model_space_delta=model_space_delta,
                    perturbation_linf=0.0,
                    perturbation_l2=0.0,
                )
            )
        return tuple(results)


class FakeLegacyClassificationAttack:
    id = "random_noise"
    supported_task_types = ["classification"]

    def __init__(self):
        self.adversarial_images: list[Image.Image] = []
        self.perturbation_images: list[Image.Image] = []

    def supports_task_type(self, task_type: str) -> bool:
        return task_type in self.supported_task_types

    def apply(
        self,
        model_wrapper,
        image: Image.Image,
        parameters,
        original_class_index: int | None = None,
    ):
        adversarial_image = image.copy()
        perturbation_image = Image.new("RGB", image.size, color=(0, 0, 0))
        self.adversarial_images.append(adversarial_image)
        self.perturbation_images.append(perturbation_image)
        return {
            "adversarial_image": adversarial_image,
            "perturbation_image": perturbation_image,
            "perturbation_linf": 0.03,
            "perturbation_l2": 1.0,
        }


def _prediction(
    probabilities: tuple[float, ...],
) -> DetailedClassificationPrediction:
    ranked_indices = sorted(
        range(len(probabilities)),
        key=lambda index: probabilities[index],
        reverse=True,
    )

    top_k = tuple(
        ClassificationClassScore(
            index=index,
            label=CLASS_LABELS[index],
            probability=probabilities[index],
        )
        for index in ranked_indices
    )

    class_index = ranked_indices[0]
    prediction_margin = (
        probabilities[ranked_indices[0]] - probabilities[ranked_indices[1]]
    )

    return DetailedClassificationPrediction(
        class_index=class_index,
        class_label=CLASS_LABELS[class_index],
        confidence=probabilities[class_index],
        probabilities=probabilities,
        top_k=top_k,
        prediction_margin=prediction_margin,
    )


def _dataset(
    labels: Sequence[str | None],
) -> DatasetResponse:
    images = [
        DatasetImage(
            id=f"image-{index}",
            filename=f"image-{index}.png",
            label=label,
        )
        for index, label in enumerate(labels, start=1)
    ]

    return DatasetResponse(
        id="dataset-1",
        name="Classification dataset",
        owner_id="user-1",
        image_count=len(images),
        images=images,
    )


async def _run_evaluation(
    monkeypatch,
    *,
    dataset: DatasetResponse,
    prediction_pairs: Sequence[
        tuple[
            DetailedClassificationPrediction,
            DetailedClassificationPrediction,
        ]
    ],
    perturbations: Sequence[tuple[float, float]] | None = None,
    source_sizes: Sequence[tuple[int, int]] | None = None,
    attack_override=None,
    attack_type: AttackType = AttackType.PGD,
    progress_callback=None,
    model_attack_micro_batch_size: int = 4,
):
    if perturbations is None:
        perturbations = [(0.03, 1.0) for _ in dataset.images]

    attack = attack_override or FakeClassificationAttack(perturbations)
    attack_batch_size = min(
        model_attack_micro_batch_size,
        getattr(attack, "evaluation_micro_batch_size", 1),
    )

    flattened_predictions = []
    for batch_start in range(0, len(prediction_pairs), attack_batch_size):
        batch_pairs = prediction_pairs[batch_start : batch_start + attack_batch_size]
        flattened_predictions.extend(pair[0] for pair in batch_pairs)
        flattened_predictions.extend(pair[1] for pair in batch_pairs)

    model = FakeClassificationModel(flattened_predictions)
    model.evaluation_attack_micro_batch_size = model_attack_micro_batch_size
    saved_metadata: dict = {}
    saved_images: dict[str, Image.Image] = {}
    saved_scientific_inputs: dict[str, dict] = {}

    if source_sizes is None:
        source_sizes = [
            (640 + index * 10, 480 + index * 5)
            for index, _ in enumerate(dataset.images, start=1)
        ]

    images_by_id = {
        image.id: Image.new(
            "RGB",
            source_size,
            color=(index, index, index),
        )
        for index, (image, source_size) in enumerate(
            zip(dataset.images, source_sizes),
            start=1,
        )
    }

    async def fake_get_dataset(dataset_id: str):
        assert dataset_id == dataset.id
        return dataset

    async def fake_get_image_from_dataset(
        dataset_id: str,
        image_id: str,
    ):
        assert dataset_id == dataset.id
        return images_by_id[image_id]

    def fake_save_result_metadata(
        result_path: str,
        metadata: dict,
    ) -> str:
        assert result_path == "/tmp/evaluation-result"
        saved_metadata.update(metadata)
        return "/tmp/evaluation-result/metadata.json"

    def fake_save_result_image(image, result_id, result_path, filename):
        saved_images[filename] = image
        return f"/results/{result_id}/{filename}"

    def fake_save_evaluation_model_inputs(**kwargs):
        saved_scientific_inputs[kwargs["image_id"]] = kwargs

    monkeypatch.setattr(
        evaluation_service,
        "get_dataset",
        fake_get_dataset,
    )
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
    monkeypatch.setattr(
        evaluation_service,
        "get_attack",
        lambda attack_type: attack,
    )
    monkeypatch.setattr(
        evaluation_service,
        "create_result_directory",
        lambda prefix: (
            "evaluation-result",
            "/tmp/evaluation-result",
        ),
    )
    monkeypatch.setattr(
        evaluation_service,
        "save_result_image",
        fake_save_result_image,
    )
    monkeypatch.setattr(
        evaluation_service,
        "save_result_metadata",
        fake_save_result_metadata,
    )
    monkeypatch.setattr(
        evaluation_service,
        "save_evaluation_model_inputs",
        fake_save_evaluation_model_inputs,
    )

    response = await evaluation_service.evaluate_dataset(
        EvaluationRequest(
            dataset_id=dataset.id,
            attack_type=attack_type,
        ),
        progress_callback=progress_callback,
    )

    if hasattr(attack, "adversarial_model_inputs"):
        adversarial_prediction_inputs = model.prepared_inputs[1::2]
        assert len(adversarial_prediction_inputs) == len(
            attack.adversarial_model_inputs
        )
        assert all(
            predicted_input is attack_input
            for predicted_input, attack_input in zip(
                adversarial_prediction_inputs,
                attack.adversarial_model_inputs,
            )
        )

    attack.saved_images = saved_images
    attack.source_images_by_id = images_by_id
    attack.saved_scientific_inputs = saved_scientific_inputs
    attack.model = model

    return response, attack, saved_metadata


@pytest.mark.asyncio
async def test_evaluation_reuses_clean_prepared_input_for_fast_path(monkeypatch):
    dataset = _dataset([None])
    attack = FastPathClassificationAttack([(0.03, 1.0)])

    await _run_evaluation(
        monkeypatch,
        dataset=dataset,
        prediction_pairs=[
            (
                _prediction((0.8, 0.1, 0.1)),
                _prediction((0.2, 0.7, 0.1)),
            )
        ],
        attack_override=attack,
        attack_type=AttackType.FGSM,
    )

    assert len(attack.model.prepared_image_inputs) == 1
    assert len(attack.model.prepared_inputs) == 2
    assert len(attack.evaluation_prepared_inputs) == 1

    prepared_for_prediction = attack.model.prepared_inputs[0]
    prepared_for_evaluation = attack.evaluation_prepared_inputs[0]
    prepared_from_image = attack.model.prepared_image_inputs[0]

    assert prepared_for_evaluation is prepared_for_prediction
    assert prepared_for_evaluation.data_ptr() == prepared_from_image.data_ptr()


@pytest.mark.asyncio
async def test_evaluation_uses_attack_micro_batches_and_preserves_result_order(
    monkeypatch,
):
    dataset = _dataset([None, None, None])
    attack = BatchFastPathClassificationAttack()
    prediction_pairs = [
        (
            _prediction((0.8, 0.1, 0.1)),
            _prediction((0.7, 0.2, 0.1)),
        ),
        (
            _prediction((0.1, 0.8, 0.1)),
            _prediction((0.2, 0.7, 0.1)),
        ),
        (
            _prediction((0.1, 0.2, 0.7)),
            _prediction((0.2, 0.2, 0.6)),
        ),
    ]

    response, _, _ = await _run_evaluation(
        monkeypatch,
        dataset=dataset,
        prediction_pairs=prediction_pairs,
        attack_override=attack,
        attack_type=AttackType.FGSM,
    )

    assert attack.batch_sizes == [2, 1]
    assert attack.original_class_index_batches == [(0, 1), (2,)]
    assert [result.image_id for result in response.results] == [
        "image-1",
        "image-2",
        "image-3",
    ]
    assert [result.original_label for result in response.results] == [
        "cat",
        "dog",
        "car",
    ]


@pytest.mark.asyncio
async def test_evaluation_keeps_single_sample_attacks_for_models_without_batch_opt_in(
    monkeypatch,
):
    dataset = _dataset([None, None])
    attack = BatchFastPathClassificationAttack()

    await _run_evaluation(
        monkeypatch,
        dataset=dataset,
        prediction_pairs=[
            (
                _prediction((0.8, 0.1, 0.1)),
                _prediction((0.7, 0.2, 0.1)),
            ),
            (
                _prediction((0.1, 0.8, 0.1)),
                _prediction((0.2, 0.7, 0.1)),
            ),
        ],
        attack_override=attack,
        attack_type=AttackType.FGSM,
        model_attack_micro_batch_size=1,
    )

    assert attack.batch_sizes == [1, 1]


@pytest.mark.parametrize("attack_type", [AttackType.FGSM, AttackType.PGD])
@pytest.mark.asyncio
async def test_evaluation_persists_each_structured_result_in_source_geometry(
    monkeypatch,
    attack_type,
):
    dataset = _dataset([None, None, None])
    source_sizes = [(19, 7), (6, 21), (13, 13)]
    prediction_pairs = [
        (
            _prediction((0.8, 0.1, 0.1)),
            _prediction((0.2, 0.7, 0.1)),
        ),
        (
            _prediction((0.1, 0.8, 0.1)),
            _prediction((0.1, 0.7, 0.2)),
        ),
        (
            _prediction((0.1, 0.2, 0.7)),
            _prediction((0.1, 0.2, 0.7)),
        ),
    ]
    projector_inputs: list[tuple[int, int]] = []
    projected_delta_refs = []
    real_reconstruct = evaluation_service.reconstruct_attack_in_source_geometry

    def tracked_reconstruct(source_image, model_space_delta):
        projector_inputs.append((id(source_image), id(model_space_delta)))
        reconstruction = real_reconstruct(source_image, model_space_delta)
        projected_delta_refs.append(ref(reconstruction.source_space_delta))
        return reconstruction

    monkeypatch.setattr(
        evaluation_service,
        "reconstruct_attack_in_source_geometry",
        tracked_reconstruct,
    )

    response, attack, saved_metadata = await _run_evaluation(
        monkeypatch,
        dataset=dataset,
        prediction_pairs=prediction_pairs,
        perturbations=[(0.03, 1.0), (0.02, 2.0), (0.01, 3.0)],
        source_sizes=source_sizes,
        attack_type=attack_type,
    )

    expected_projector_inputs = [
        (id(attack.source_images_by_id[image.id]), id(model_space_delta))
        for image, model_space_delta in zip(
            dataset.images,
            attack.model_space_deltas,
        )
    ]
    assert projector_inputs == expected_projector_inputs
    assert all(
        projected_delta_ref() is None for projected_delta_ref in projected_delta_refs
    )

    for index, (image_metadata, source_size) in enumerate(
        zip(dataset.images, source_sizes)
    ):
        adversarial_filename = f"{image_metadata.id}_adversarial.png"
        perturbation_filename = f"{image_metadata.id}_perturbation.png"
        saved_adversarial = attack.saved_images[adversarial_filename]
        saved_perturbation = attack.saved_images[perturbation_filename]

        assert saved_adversarial.size == source_size
        assert saved_perturbation.size == source_size
        assert saved_adversarial.mode == "RGB"
        assert saved_perturbation.mode == "RGB"
        assert saved_adversarial is not attack.temporary_adversarial_images[index]
        assert saved_perturbation is not attack.temporary_perturbation_images[index]
        assert response.results[index].adversarial_image_url.endswith(
            f"/{adversarial_filename}"
        )
        assert response.results[index].perturbation_image_url.endswith(
            f"/{perturbation_filename}"
        )
        scientific_inputs = attack.saved_scientific_inputs[image_metadata.id]
        assert (
            scientific_inputs["clean_model_input"]
            is attack.model.prepared_inputs[index * 2]
        )
        assert (
            scientific_inputs["adversarial_model_input"]
            is attack.adversarial_model_inputs[index]
        )
        manifest_entry = saved_metadata["images"][image_metadata.id]
        assert manifest_entry["image_id"] == image_metadata.id
        assert manifest_entry["gradcam_generated"] is False
        assert manifest_entry["adversarial_visual_filename"] == adversarial_filename

    assert [result.perturbation_linf for result in response.results] == [
        0.03,
        0.02,
        0.01,
    ]
    assert [result.perturbation_l2 for result in response.results] == [1.0, 2.0, 3.0]
    assert response.summary.average_perturbation_linf == 0.02
    assert response.summary.average_perturbation_l2 == 2.0


@pytest.mark.asyncio
async def test_random_noise_evaluation_keeps_legacy_visual_flow(monkeypatch):
    dataset = _dataset([None])
    attack = FakeLegacyClassificationAttack()

    def fail_if_reconstruction_is_used(source_image, model_space_delta):
        raise AssertionError("Random Noise must not use source reconstruction")

    monkeypatch.setattr(
        evaluation_service,
        "reconstruct_attack_in_source_geometry",
        fail_if_reconstruction_is_used,
    )

    response, attack, _ = await _run_evaluation(
        monkeypatch,
        dataset=dataset,
        prediction_pairs=[
            (
                _prediction((0.8, 0.1, 0.1)),
                _prediction((0.2, 0.7, 0.1)),
            )
        ],
        source_sizes=[(17, 9)],
        attack_override=attack,
        attack_type=AttackType.RANDOM_NOISE,
    )

    assert (
        attack.saved_images["image-1_adversarial.png"] is attack.adversarial_images[0]
    )
    assert (
        attack.saved_images["image-1_perturbation.png"] is attack.perturbation_images[0]
    )
    scientific_inputs = attack.saved_scientific_inputs["image-1"]
    assert scientific_inputs["clean_model_input"] is attack.model.prepared_inputs[0]
    assert (
        scientific_inputs["adversarial_model_input"] is attack.model.prepared_inputs[1]
    )
    assert response.results[0].perturbation_linf == 0.03
    assert response.results[0].perturbation_l2 == 1.0
    assert [item.label for item in response.results[0].original_top_k] == [
        "cat",
        "dog",
        "car",
    ]
    assert [item.probability for item in response.results[0].original_top_k] == [
        pytest.approx(0.8),
        pytest.approx(0.1),
        pytest.approx(0.1),
    ]
    assert [item.label for item in response.results[0].adversarial_top_k] == [
        "dog",
        "cat",
        "car",
    ]


@pytest.mark.asyncio
async def test_evaluation_exposes_complete_advanced_metrics(
    monkeypatch,
):
    dataset = _dataset(["cat", "dog", "car", "cat"])

    prediction_pairs = [
        (
            _prediction((0.8, 0.1, 0.1)),
            _prediction((0.2, 0.7, 0.1)),
        ),
        (
            _prediction((0.1, 0.8, 0.1)),
            _prediction((0.1, 0.7, 0.2)),
        ),
        (
            _prediction((0.1, 0.8, 0.1)),
            _prediction((0.1, 0.2, 0.7)),
        ),
        (
            _prediction((0.1, 0.2, 0.7)),
            _prediction((0.1, 0.3, 0.6)),
        ),
    ]

    response, attack, saved_metadata = await _run_evaluation(
        monkeypatch,
        dataset=dataset,
        prediction_pairs=prediction_pairs,
        perturbations=[
            (0.03, 1.0),
            (0.02, 2.0),
            (0.01, 3.0),
            (0.02, 4.0),
        ],
    )

    metrics = response.advanced_metrics
    robustness = metrics.robustness
    supervised = metrics.supervised

    assert metrics.metrics_schema_version == "1.0"
    assert metrics.delta_convention == "original_minus_adversarial"
    assert metrics.supervised_status.available is True
    assert supervised is not None

    assert robustness.total_images == 4
    assert robustness.changed_predictions == 2
    assert robustness.unchanged_predictions == 2
    assert robustness.prediction_flip_rate == 0.5
    assert robustness.correct_to_incorrect == 1
    assert robustness.incorrect_to_correct == 1
    assert robustness.clean_accuracy.value == 0.5
    assert robustness.robust_accuracy.value == 0.5
    assert robustness.accuracy_drop.value == 0.0
    assert robustness.conditional_attack_success_rate.value == 0.5
    assert robustness.corrected_by_attack_rate.value == 0.5
    assert robustness.average_perturbation_linf == 0.02
    assert robustness.average_perturbation_l2 == 2.5

    summary = response.summary

    assert summary.total_images == robustness.total_images
    assert summary.changed_predictions == robustness.changed_predictions
    assert summary.unchanged_predictions == robustness.unchanged_predictions
    assert summary.attack_success_rate == robustness.prediction_flip_rate
    assert summary.original_accuracy == robustness.clean_accuracy.value
    assert summary.adversarial_accuracy == robustness.robust_accuracy.value
    assert summary.accuracy_drop == robustness.accuracy_drop.value
    assert summary.correct_before_wrong_after == robustness.correct_to_incorrect

    aggregate = supervised.aggregate

    assert aggregate.delta.accuracy == (
        aggregate.original.accuracy - aggregate.adversarial.accuracy
    )

    matrices = supervised.confusion_matrices
    matrix_size = len(matrices.labels)

    assert matrix_size == 3
    assert all(
        len(matrix) == matrix_size and all(len(row) == matrix_size for row in matrix)
        for matrix in (
            matrices.original,
            matrices.adversarial,
            matrices.original_normalized,
            matrices.adversarial_normalized,
        )
    )

    assert supervised.roc.micro.original.available is True
    assert supervised.precision_recall.micro.original.available is True
    assert supervised.roc.micro.original.points[0].threshold is None
    assert supervised.precision_recall.micro.original.points[-1].threshold is None

    transitions = [result.diagnostics.transition for result in response.results]

    assert transitions == [
        "correct_to_incorrect",
        "correct_to_correct",
        "incorrect_to_correct",
        "incorrect_to_incorrect",
    ]

    first_diagnostics = response.results[0].diagnostics

    assert first_diagnostics.original_true_class_probability == pytest.approx(0.8)
    assert first_diagnostics.adversarial_true_class_probability == pytest.approx(0.2)
    assert first_diagnostics.true_class_probability_drop == pytest.approx(0.6)
    assert first_diagnostics.original_top_5_correct is True
    assert first_diagnostics.adversarial_top_5_correct is True

    assert attack.original_class_indices == [0, 1, 1, 2]

    expected_metadata = {
        "result_id": "evaluation-result",
        "result_type": "evaluation",
        "dataset_id": "dataset-1",
        "model_id": "test-classifier",
        "attack_type": AttackType.PGD,
        "parameters": {
            "epsilon": 0.03,
            "alpha": 0.01,
            "steps": 10,
            "seed": 0,
        },
        "total_images": 4,
        "resolved_labels": 4,
        "unresolved_labels": 0,
        "metrics_schema_version": "1.0",
        "expires_in_hours": 24,
    }
    assert {key: saved_metadata[key] for key in expected_metadata} == expected_metadata
    assert saved_metadata["status"] == "completed"
    assert saved_metadata["owner_id"] == "user-1"
    assert set(saved_metadata["images"]) == {
        "image-1",
        "image-2",
        "image-3",
        "image-4",
    }


@pytest.mark.asyncio
async def test_evaluation_excludes_only_unresolved_labels_from_supervised_metrics(
    monkeypatch,
):
    dataset = _dataset(["cat", "unknown", None])

    response, _, _ = await _run_evaluation(
        monkeypatch,
        dataset=dataset,
        prediction_pairs=[
            (
                _prediction((0.8, 0.1, 0.1)),
                _prediction((0.2, 0.7, 0.1)),
            ),
            (
                _prediction((0.1, 0.8, 0.1)),
                _prediction((0.1, 0.2, 0.7)),
            ),
            (
                _prediction((0.1, 0.2, 0.7)),
                _prediction((0.1, 0.2, 0.7)),
            ),
        ],
    )

    metrics = response.advanced_metrics
    label_resolution = metrics.label_resolution
    robustness = metrics.robustness
    supervised = metrics.supervised

    assert label_resolution.labeled_images == 2
    assert label_resolution.resolved_labels == 1
    assert label_resolution.unresolved_labels == 1
    assert label_resolution.coverage == 0.5
    assert label_resolution.unresolved[0].image_id == "image-2"

    assert robustness.total_images == 3
    assert robustness.changed_predictions == 2
    assert robustness.unchanged_predictions == 1

    assert metrics.supervised_status.available is True
    assert "Imágenes etiquetadas excluidas por etiquetas no resueltas: 1" in metrics.supervised_status.warning
    assert supervised is not None
    assert supervised.aggregate.original.accuracy == 1.0
    assert supervised.aggregate.adversarial.accuracy == 0.0

    assert supervised.roc.micro.original.available is False
    assert "muestra negativa" in supervised.roc.micro.original.reason
    assert supervised.precision_recall.micro.original.available is True

    class_curve = supervised.precision_recall.per_class[0]

    assert class_curve.support == 1
    assert "menos de 2 muestras positivas" in class_curve.curves.original.warning

    unresolved_diagnostics = response.results[1].diagnostics
    unlabeled_diagnostics = response.results[2].diagnostics

    assert unresolved_diagnostics.ground_truth_class is None
    assert unresolved_diagnostics.transition is None
    assert (
        unresolved_diagnostics.label_resolution_reason
        == "La etiqueta no coincide con ninguna clase del modelo"
    )

    assert unlabeled_diagnostics.ground_truth_class is None
    assert unlabeled_diagnostics.transition is None
    assert (
        unlabeled_diagnostics.label_resolution_reason
        == "La imagen no tiene una etiqueta de referencia"
    )

    assert response.summary.labeled_images == 1
    assert response.summary.original_accuracy == 1.0
    assert response.summary.adversarial_accuracy == 0.0


@pytest.mark.asyncio
async def test_evaluation_without_labels_keeps_unsupervised_robustness(
    monkeypatch,
):
    dataset = _dataset([None, None])

    response, _, _ = await _run_evaluation(
        monkeypatch,
        dataset=dataset,
        prediction_pairs=[
            (
                _prediction((0.8, 0.1, 0.1)),
                _prediction((0.2, 0.7, 0.1)),
            ),
            (
                _prediction((0.1, 0.8, 0.1)),
                _prediction((0.1, 0.8, 0.1)),
            ),
        ],
    )

    metrics = response.advanced_metrics
    robustness = metrics.robustness

    assert metrics.label_resolution.labeled_images == 0
    assert metrics.label_resolution.resolved_labels == 0
    assert metrics.supervised is None
    assert metrics.supervised_status.available is False
    assert metrics.supervised_status.reason == "El conjunto de datos no contiene imágenes etiquetadas"

    assert robustness.total_images == 2
    assert robustness.changed_predictions == 1
    assert robustness.unchanged_predictions == 1
    assert robustness.prediction_flip_rate == 0.5
    assert robustness.clean_accuracy.available is False
    assert robustness.clean_accuracy.reason == "El conjunto de datos no contiene imágenes etiquetadas"

    assert response.summary.labeled_images == 0
    assert response.summary.original_accuracy is None
    assert response.summary.adversarial_accuracy is None
    assert response.summary.accuracy_drop is None

    assert all(
        result.diagnostics.transition is None
        and result.diagnostics.label_resolution_reason
        == "La imagen no tiene una etiqueta de referencia"
        for result in response.results
    )


@pytest.mark.asyncio
async def test_evaluation_with_only_unresolved_labels_disables_supervised_metrics(
    monkeypatch,
):
    dataset = _dataset(["unknown"])

    response, _, _ = await _run_evaluation(
        monkeypatch,
        dataset=dataset,
        prediction_pairs=[
            (
                _prediction((0.8, 0.1, 0.1)),
                _prediction((0.2, 0.7, 0.1)),
            ),
        ],
    )

    metrics = response.advanced_metrics

    assert metrics.label_resolution.labeled_images == 1
    assert metrics.label_resolution.resolved_labels == 0
    assert metrics.label_resolution.unresolved_labels == 1
    assert metrics.label_resolution.coverage == 0.0

    assert metrics.supervised is None
    assert metrics.supervised_status.available is False
    assert (
        metrics.supervised_status.reason
        == "Ninguna etiqueta del conjunto de datos coincide con las clases del modelo activo"
    )

    assert metrics.robustness.changed_predictions == 1
    assert metrics.robustness.prediction_flip_rate == 1.0
    assert metrics.robustness.clean_accuracy.available is False

    diagnostics = response.results[0].diagnostics

    assert diagnostics.ground_truth_class is None
    assert diagnostics.transition is None
    assert diagnostics.label_resolution_reason == "La etiqueta no coincide con ninguna clase del modelo"

    assert response.summary.labeled_images == 0
    assert response.summary.original_accuracy is None


@pytest.mark.asyncio
async def test_classification_evaluation_reports_real_image_progress(monkeypatch):
    dataset = _dataset([None, None])
    progress_events = []

    await _run_evaluation(
        monkeypatch,
        dataset=dataset,
        prediction_pairs=[
            (_prediction((0.8, 0.1, 0.1)), _prediction((0.7, 0.2, 0.1))),
            (_prediction((0.1, 0.8, 0.1)), _prediction((0.2, 0.7, 0.1))),
        ],
        progress_callback=progress_events.append,
    )

    assert [
        (event.processed, event.total, event.phase) for event in progress_events
    ] == [
        (0, 2, "images"),
        (1, 2, "images"),
        (2, 2, "images"),
        (2, 2, "metrics"),
    ]

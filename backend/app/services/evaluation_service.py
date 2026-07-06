import logging
import sys
from time import perf_counter

import torch

from core.errors import BadRequestError
from core.task_compatibility import (
    CLASSIFICATION_TASK_TYPE,
    require_attack_task_type,
    require_model_task_type,
)
from ml.attacks.registry import get_attack
from ml.metrics.classification import (
    ClassificationEvaluationSample,
    build_evaluation_image_diagnostics,
    calculate_advanced_classification_metrics,
)
from schemas.evaluation import (
    EvaluationImageResult,
    EvaluationRequest,
    EvaluationResponse,
    EvaluationSummary,
)
from services.classification_label_service import (
    resolve_dataset_labels,
)
from services.dataset_service import (
    get_dataset,
    get_image_from_dataset,
)
from services.model_service import get_active_model_instance
from utils.result_files import (
    create_result_directory,
    save_result_image,
    save_result_metadata,
)


logger = logging.getLogger("classification_evaluation")
logger.setLevel(logging.INFO)
logger.propagate = False

if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s "
            "%(name)s %(message)s"
        )
    )
    logger.addHandler(handler)


def _get_model_device(model) -> torch.device | None:
    """Return the underlying torch device when the model exposes one."""
    get_torch_model = getattr(model, "get_torch_model", None)

    if not callable(get_torch_model):
        return None

    try:
        torch_model = get_torch_model()
        return next(torch_model.parameters()).device
    except (
        AttributeError,
        NotImplementedError,
        StopIteration,
        TypeError,
        ValueError,
    ):
        return None


def _synchronize_device(device: torch.device | None) -> None:
    """Synchronize CUDA so timings include completed GPU work."""
    if device is not None and device.type == "cuda":
        torch.cuda.synchronize(device)


def _start_stage(device: torch.device | None = None) -> float:
    _synchronize_device(device)
    return perf_counter()


def _finish_stage(
    started_at: float,
    device: torch.device | None = None,
) -> float:
    _synchronize_device(device)
    return perf_counter() - started_at


async def evaluate_dataset(
    request: EvaluationRequest,
) -> EvaluationResponse:
    logger.debug(
        (
            "classification_evaluation_started "
            "dataset_id=%s attack=%s steps=%s"
        ),
        request.dataset_id,
        request.attack_type,
        getattr(request.parameters, "steps", None),
    )

    evaluation_started = perf_counter()

    dataset_started = perf_counter()
    dataset = await get_dataset(request.dataset_id)
    dataset_load_seconds = perf_counter() - dataset_started

    if not dataset.images:
        raise BadRequestError(
            "Dataset must contain at least one image"
        )

    setup_started = perf_counter()

    model = get_active_model_instance()

    require_model_task_type(
        model=model,
        expected_task_type=CLASSIFICATION_TASK_TYPE,
        endpoint="/evaluate",
    )

    try:
        attack = get_attack(request.attack_type)
    except ValueError as error:
        raise BadRequestError(str(error)) from error

    require_attack_task_type(
        attack=attack,
        task_type=model.task_type,
    )

    label_resolution = resolve_dataset_labels(
        dataset=dataset,
        model=model,
        endpoint="/evaluate",
    )

    resolved_by_image_id = (
        label_resolution.resolved_by_image_id
    )

    unresolved_by_image_id = {
        unresolved.image_id: unresolved
        for unresolved in label_resolution.summary.unresolved
    }

    result_id, result_path = create_result_directory(
        prefix="evaluation"
    )

    setup_seconds = perf_counter() - setup_started
    model_device = _get_model_device(model)
    device_name = (
        str(model_device)
        if model_device is not None
        else "unknown"
    )

    profile_totals = {
        "load_image": 0.0,
        "original_prediction": 0.0,
        "attack": 0.0,
        "adversarial_prediction": 0.0,
        "save_images": 0.0,
    }

    results: list[EvaluationImageResult] = []
    metric_samples: list[ClassificationEvaluationSample] = []

    for image_metadata in dataset.images:
        image_started = perf_counter()

        image_id = image_metadata.id
        filename = image_metadata.filename

        stage_started = perf_counter()
        image = await get_image_from_dataset(
            dataset_id=request.dataset_id,
            image_id=image_id,
        )
        load_image_seconds = perf_counter() - stage_started
        profile_totals["load_image"] += load_image_seconds

        image_width, image_height = image.size

        stage_started = _start_stage(model_device)
        original_prediction = model.predict_detailed(
            image=image,
            top_k=5,
        )
        original_prediction_seconds = _finish_stage(
            stage_started,
            model_device,
        )
        profile_totals[
            "original_prediction"
        ] += original_prediction_seconds

        stage_started = _start_stage(model_device)
        attack_result = attack.apply(
            model_wrapper=model,
            image=image,
            parameters=request.parameters,
            original_class_index=(
                original_prediction.class_index
            ),
        )
        attack_seconds = _finish_stage(
            stage_started,
            model_device,
        )
        profile_totals["attack"] += attack_seconds

        adversarial_image = attack_result[
            "adversarial_image"
        ]
        perturbation_image = attack_result[
            "perturbation_image"
        ]
        attack_width, attack_height = (
            adversarial_image.size
        )

        stage_started = _start_stage(model_device)
        adversarial_prediction = model.predict_detailed(
            image=adversarial_image,
            top_k=5,
        )
        adversarial_prediction_seconds = _finish_stage(
            stage_started,
            model_device,
        )
        profile_totals[
            "adversarial_prediction"
        ] += adversarial_prediction_seconds

        resolved_label = resolved_by_image_id.get(image_id)

        ground_truth = (
            resolved_label.class_reference
            if resolved_label is not None
            else None
        )

        unresolved_label = unresolved_by_image_id.get(
            image_id
        )

        label_resolution_reason = None

        if ground_truth is None:
            if unresolved_label is not None:
                label_resolution_reason = (
                    unresolved_label.reason
                )
            else:
                label_resolution_reason = (
                    "Image has no ground-truth label"
                )

        diagnostics = build_evaluation_image_diagnostics(
            ground_truth=ground_truth,
            original=original_prediction,
            adversarial=adversarial_prediction,
            label_resolution_reason=(
                label_resolution_reason
            ),
        )

        original_correct = None
        adversarial_correct = None

        if ground_truth is not None:
            original_correct = (
                original_prediction.class_index
                == ground_truth.index
            )
            adversarial_correct = (
                adversarial_prediction.class_index
                == ground_truth.index
            )

        prediction_changed = (
            original_prediction.class_index
            != adversarial_prediction.class_index
        )

        original_confidence = round(
            original_prediction.confidence,
            4,
        )

        adversarial_confidence = round(
            adversarial_prediction.confidence,
            4,
        )

        confidence_difference = round(
            adversarial_prediction.confidence
            - original_prediction.confidence,
            4,
        )

        safe_image_id = image_id.replace("/", "_")

        stage_started = perf_counter()

        adversarial_image_url = save_result_image(
            image=adversarial_image,
            result_id=result_id,
            result_path=result_path,
            filename=(
                f"{safe_image_id}_adversarial.png"
            ),
        )

        perturbation_image_url = save_result_image(
            image=perturbation_image,
            result_id=result_id,
            result_path=result_path,
            filename=(
                f"{safe_image_id}_perturbation.png"
            ),
        )

        save_images_seconds = perf_counter() - stage_started
        profile_totals["save_images"] += save_images_seconds

        original_image_url = (
            f"/datasets/{request.dataset_id}"
            f"/images/{image_id}"
        )

        results.append(
            EvaluationImageResult(
                image_id=image_id,
                filename=filename,
                original_image_url=original_image_url,
                adversarial_image_url=(
                    adversarial_image_url
                ),
                perturbation_image_url=(
                    perturbation_image_url
                ),
                ground_truth_label=image_metadata.label,
                original_label=(
                    original_prediction.class_label
                ),
                adversarial_label=(
                    adversarial_prediction.class_label
                ),
                original_confidence=original_confidence,
                adversarial_confidence=(
                    adversarial_confidence
                ),
                confidence_difference=(
                    confidence_difference
                ),
                prediction_changed=prediction_changed,
                # Legacy field. The new UI uses
                # diagnostics.transition and Conditional ASR.
                attack_success=prediction_changed,
                original_correct=original_correct,
                adversarial_correct=adversarial_correct,
                perturbation_linf=attack_result[
                    "perturbation_linf"
                ],
                perturbation_l2=attack_result[
                    "perturbation_l2"
                ],
                diagnostics=diagnostics,
            )
        )

        metric_samples.append(
            ClassificationEvaluationSample(
                ground_truth=ground_truth,
                original=original_prediction,
                adversarial=adversarial_prediction,
                perturbation_linf=attack_result[
                    "perturbation_linf"
                ],
                perturbation_l2=attack_result[
                    "perturbation_l2"
                ],
            )
        )

        image_total_seconds = perf_counter() - image_started

        logger.debug(
            (
                "classification_evaluation_image_profile "
                "image_id=%s filename=%s "
                "source_size=%sx%s attack_size=%sx%s "
                "load_image=%.4fs "
                "original_prediction=%.4fs "
                "attack=%.4fs "
                "adversarial_prediction=%.4fs "
                "save_images=%.4fs total=%.4fs"
            ),
            image_id,
            filename,
            image_width,
            image_height,
            attack_width,
            attack_height,
            load_image_seconds,
            original_prediction_seconds,
            attack_seconds,
            adversarial_prediction_seconds,
            save_images_seconds,
            image_total_seconds,
        )

    metrics_started = perf_counter()
    advanced_metrics = (
        calculate_advanced_classification_metrics(
            samples=metric_samples,
            label_resolution=label_resolution.summary,
            class_labels=model.get_class_labels(),
        )
    )
    metrics_seconds = perf_counter() - metrics_started

    total_images = len(results)

    average_original_confidence = round(
        sum(
            result.original_confidence
            for result in results
        )
        / total_images,
        4,
    )

    average_adversarial_confidence = round(
        sum(
            result.adversarial_confidence
            for result in results
        )
        / total_images,
        4,
    )

    average_confidence_difference = round(
        sum(
            result.confidence_difference
            for result in results
        )
        / total_images,
        4,
    )

    resolved_results = [
        result
        for result in results
        if result.original_correct is not None
    ]

    correct_before_attack = sum(
        result.original_correct is True
        for result in resolved_results
    )

    correct_after_attack = sum(
        result.adversarial_correct is True
        for result in resolved_results
    )

    robustness = advanced_metrics.robustness

    summary = EvaluationSummary(
        total_images=total_images,
        changed_predictions=(
            robustness.changed_predictions
        ),
        unchanged_predictions=(
            robustness.unchanged_predictions
        ),
        attack_success_rate=(
            robustness.prediction_flip_rate
        ),
        average_original_confidence=(
            average_original_confidence
        ),
        average_adversarial_confidence=(
            average_adversarial_confidence
        ),
        average_confidence_difference=(
            average_confidence_difference
        ),
        average_perturbation_linf=(
            robustness.average_perturbation_linf
        ),
        average_perturbation_l2=(
            robustness.average_perturbation_l2
        ),
        # Legacy summary uses only resolved labels so the
        # denominator remains coherent.
        labeled_images=(
            label_resolution.summary.resolved_labels
        ),
        original_accuracy=(
            robustness.clean_accuracy.value
            if robustness.clean_accuracy.available
            else None
        ),
        adversarial_accuracy=(
            robustness.robust_accuracy.value
            if robustness.robust_accuracy.available
            else None
        ),
        accuracy_drop=(
            robustness.accuracy_drop.value
            if robustness.accuracy_drop.available
            else None
        ),
        correct_before_attack=(
            correct_before_attack
            if resolved_results
            else None
        ),
        correct_after_attack=(
            correct_after_attack
            if resolved_results
            else None
        ),
        correct_before_wrong_after=(
            robustness.correct_to_incorrect
            if resolved_results
            else None
        ),
    )

    metadata_started = perf_counter()
    save_result_metadata(
        result_path=result_path,
        metadata={
            "result_id": result_id,
            "result_type": "evaluation",
            "dataset_id": request.dataset_id,
            "model_id": model.id,
            "attack_type": request.attack_type,
            "parameters": request.parameters.model_dump(),
            "total_images": total_images,
            "resolved_labels": (
                label_resolution.summary.resolved_labels
            ),
            "unresolved_labels": (
                label_resolution.summary.unresolved_labels
            ),
            "metrics_schema_version": (
                advanced_metrics.metrics_schema_version
            ),
            "expires_in_hours": 24,
        },
    )
    metadata_seconds = perf_counter() - metadata_started

    total_seconds = perf_counter() - evaluation_started
    attack_steps = getattr(
        request.parameters,
        "steps",
        None,
    )

    logger.info(
        (
            "classification_evaluation_profile "
            "dataset_id=%s attack=%s steps=%s "
            "images=%d device=%s "
            "dataset_load=%.4fs setup=%.4fs "
            "load_images=%.4fs "
            "original_predictions=%.4fs "
            "attack_total=%.4fs "
            "adversarial_predictions=%.4fs "
            "save_images=%.4fs metrics=%.4fs "
            "metadata=%.4fs total=%.4fs"
        ),
        request.dataset_id,
        request.attack_type,
        attack_steps,
        total_images,
        device_name,
        dataset_load_seconds,
        setup_seconds,
        profile_totals["load_image"],
        profile_totals["original_prediction"],
        profile_totals["attack"],
        profile_totals["adversarial_prediction"],
        profile_totals["save_images"],
        metrics_seconds,
        metadata_seconds,
        total_seconds,
    )

    return EvaluationResponse(
        result_id=result_id,
        dataset_id=request.dataset_id,
        model_id=model.id,
        model_name=model.name,
        attack_type=request.attack_type,
        parameters=request.parameters,
        summary=summary,
        advanced_metrics=advanced_metrics,
        results=results,
    )
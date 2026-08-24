import asyncio
import logging
import sys
from datetime import datetime, timedelta, timezone
from time import perf_counter

from core.cancellation import CancellationToken
from core.errors import BadRequestError
from core.task_compatibility import (
    CLASSIFICATION_TASK_TYPE,
    require_attack_task_type,
    require_model_task_type,
)
from ml.attacks.base import (
    AdversarialAttackResult,
    AdversarialEvaluationResult,
)
from ml.attacks.presentation import reconstruct_attack_in_source_geometry
from ml.attacks.registry import (
    apply_attack_runner_batch_for_evaluation,
    create_attack_runner,
    get_attack,
    get_attack_runner_evaluation_batch_size,
    resolve_attack_parameters,
)
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
    dataset_read_snapshot,
    get_dataset,
    get_image_from_dataset,
)
from services.evaluation_history_service import (
    EvaluationExecution,
    abort_evaluation_execution,
    begin_evaluation_execution,
    complete_evaluation_execution,
)
from services.evaluation_profiling import (
    EvaluationProfile,
    finish_profile_stage,
    get_model_device,
    log_evaluation_profile,
    start_profile_stage,
)
from services.evaluation_progress import (
    EvaluationProgressCallback,
    report_evaluation_progress,
)
from services.model_service import get_active_model_instance
from utils.evaluation_artifacts import (
    evaluation_image_artifact_filenames,
    save_evaluation_model_inputs,
)
from utils.reproducibility import attack_parameters_for_image
from utils.result_files import (
    RESULTS_TTL_SECONDS,
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
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    logger.addHandler(handler)


async def evaluate_dataset(
    request: EvaluationRequest,
    *,
    owner_id: str | None = None,
    retention_seconds: int = RESULTS_TTL_SECONDS,
    progress_callback: EvaluationProgressCallback | None = None,
    cancellation_token: CancellationToken | None = None,
) -> EvaluationResponse:
    try:
        attack = get_attack(request.attack_type)
        request.parameters = resolve_attack_parameters(attack, request.parameters)
    except ValueError as error:
        raise BadRequestError(str(error)) from error

    execution = None
    if owner_id is not None:
        execution = await begin_evaluation_execution(
            request=request,
            owner_id=owner_id,
            task_type="classification",
            retention_seconds=retention_seconds,
        )

    completed = False
    try:
        with dataset_read_snapshot(request.dataset_id):
            response = await _evaluate_dataset(
                request,
                attack=attack,
                execution=execution,
                progress_callback=progress_callback,
                cancellation_token=cancellation_token,
            )
        completed = True
        return response
    finally:
        if execution is not None and not completed:
            await abort_evaluation_execution(execution)


async def _iter_attacked_evaluation_images(
    *,
    dataset,
    request: EvaluationRequest,
    model,
    attack_runner,
    model_device: str,
    profile_totals: dict[str, float],
    cancellation_token: CancellationToken | None,
):
    """Prepare inputs and execute compatible attacks in bounded micro-batches."""
    model_attack_batch_size = getattr(
        model,
        "evaluation_attack_micro_batch_size",
        1,
    )
    if isinstance(model_attack_batch_size, bool) or not isinstance(
        model_attack_batch_size,
        int,
    ):
        raise TypeError("evaluation_attack_micro_batch_size must be an integer")
    if model_attack_batch_size < 1:
        raise ValueError("evaluation_attack_micro_batch_size must be positive")

    attack_batch_size = min(
        model_attack_batch_size,
        get_attack_runner_evaluation_batch_size(attack_runner),
    )

    for batch_start in range(0, len(dataset.images), attack_batch_size):
        batch_metadata = dataset.images[batch_start : batch_start + attack_batch_size]
        prepared_batch = []

        for batch_offset, image_metadata in enumerate(batch_metadata):
            if cancellation_token is not None:
                cancellation_token.raise_if_cancelled()

            image_index = batch_start + batch_offset + 1
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

            stage_started = start_profile_stage(model_device)
            clean_model_input = model.prepare_attack_input(image).detach()
            original_prediction = model.predict_detailed_from_prepared_input(
                prepared_input=clean_model_input,
                top_k=5,
            )
            original_prediction_seconds = finish_profile_stage(
                stage_started,
                model_device,
            )
            profile_totals["original_prediction"] += original_prediction_seconds

            prepared_batch.append(
                {
                    "image_index": image_index,
                    "image_metadata": image_metadata,
                    "image_id": image_id,
                    "filename": filename,
                    "image": image,
                    "image_width": image_width,
                    "image_height": image_height,
                    "load_image_seconds": load_image_seconds,
                    "clean_model_input": clean_model_input,
                    "original_prediction": original_prediction,
                    "original_prediction_seconds": original_prediction_seconds,
                    "parameters": attack_parameters_for_image(
                        attack_type=request.attack_type,
                        parameters=request.parameters,
                        image_id=image_id,
                    ),
                }
            )

        stage_started = start_profile_stage(model_device)
        if cancellation_token is not None:
            cancellation_token.raise_if_cancelled()
        attack_results = await asyncio.to_thread(
            apply_attack_runner_batch_for_evaluation,
            attack_runner,
            model_wrapper=model,
            images=tuple(item["image"] for item in prepared_batch),
            prepared_inputs=tuple(item["clean_model_input"] for item in prepared_batch),
            parameters=tuple(item["parameters"] for item in prepared_batch),
            original_class_indices=tuple(
                item["original_prediction"].class_index for item in prepared_batch
            ),
        )
        if cancellation_token is not None:
            cancellation_token.raise_if_cancelled()
        attack_batch_seconds = finish_profile_stage(
            stage_started,
            model_device,
        )
        profile_totals["attack"] += attack_batch_seconds
        attack_seconds_per_image = attack_batch_seconds / len(prepared_batch)

        for prepared_item, attack_result in zip(prepared_batch, attack_results):
            yield {
                **prepared_item,
                "attack_result": attack_result,
                "attack_seconds": attack_seconds_per_image,
                "attack_batch_size": len(prepared_batch),
            }


async def _evaluate_dataset(
    request: EvaluationRequest,
    *,
    attack,
    execution: EvaluationExecution | None,
    progress_callback: EvaluationProgressCallback | None,
    cancellation_token: CancellationToken | None,
) -> EvaluationResponse:
    logger.debug(
        ("classification_evaluation_started dataset_id=%s attack=%s steps=%s"),
        request.dataset_id,
        request.attack_type,
        request.parameters.model_dump().get("steps"),
    )

    profile = EvaluationProfile(
        task_type="classification",
        dataset_id=request.dataset_id,
        attack_type=request.attack_type.value,
    )

    dataset_started = perf_counter()
    dataset = await get_dataset(request.dataset_id)
    dataset_load_seconds = perf_counter() - dataset_started
    profile.add("dataset_load", dataset_load_seconds)

    if not dataset.images:
        raise BadRequestError("Dataset must contain at least one image")

    setup_started = perf_counter()

    model = get_active_model_instance()

    require_model_task_type(
        model=model,
        expected_task_type=CLASSIFICATION_TASK_TYPE,
        endpoint="/evaluate",
    )

    require_attack_task_type(
        attack=attack,
        task_type=model.task_type,
    )

    label_resolution = resolve_dataset_labels(
        dataset=dataset,
        model=model,
        endpoint="/evaluate",
    )

    resolved_by_image_id = label_resolution.resolved_by_image_id

    unresolved_by_image_id = {
        unresolved.image_id: unresolved
        for unresolved in label_resolution.summary.unresolved
    }

    if execution is None:
        result_id, result_path = create_result_directory(prefix="evaluation")
    else:
        result_id, result_path = create_result_directory(
            prefix="evaluation",
            result_id=execution.result_id,
        )
    result_retention_seconds = (
        execution.retention_seconds if execution is not None else RESULTS_TTL_SECONDS
    )
    dataset_revision = (
        execution.dataset_revision
        if execution is not None
        else int(getattr(dataset, "dataset_revision", 0))
    )
    evaluation_images_manifest: dict[str, dict] = {}

    save_result_metadata(
        result_path=result_path,
        metadata={
            "result_id": result_id,
            "result_type": "evaluation",
            "status": "processing",
            "owner_id": str(dataset.owner_id),
            "dataset_id": request.dataset_id,
            "model_id": model.id,
            "attack_type": request.attack_type,
            "parameters": request.parameters.model_dump(),
            "images": evaluation_images_manifest,
            "retention_seconds": result_retention_seconds,
            "expires_in_hours": result_retention_seconds // 3600,
        },
    )

    attack_runner = create_attack_runner(
        attack=attack,
        model_wrapper=model,
        parameters=request.parameters,
        cancellation_token=cancellation_token,
    )

    setup_seconds = perf_counter() - setup_started
    profile.add("setup", setup_seconds)
    model_device = get_model_device(model)
    profile_totals = {
        "load_image": 0.0,
        "original_prediction": 0.0,
        "attack": 0.0,
        "adversarial_prediction": 0.0,
        "save_images": 0.0,
    }

    results: list[EvaluationImageResult] = []
    metric_samples: list[ClassificationEvaluationSample] = []
    progress_total = len(dataset.images)
    report_evaluation_progress(
        progress_callback,
        processed=0,
        total=progress_total,
    )

    async for prepared_item in _iter_attacked_evaluation_images(
        dataset=dataset,
        request=request,
        model=model,
        attack_runner=attack_runner,
        model_device=model_device,
        profile_totals=profile_totals,
        cancellation_token=cancellation_token,
    ):
        image_index = prepared_item["image_index"]
        image_metadata = prepared_item["image_metadata"]
        image_id = prepared_item["image_id"]
        filename = prepared_item["filename"]
        image = prepared_item["image"]
        image_width = prepared_item["image_width"]
        image_height = prepared_item["image_height"]
        load_image_seconds = prepared_item["load_image_seconds"]
        clean_model_input = prepared_item["clean_model_input"]
        original_prediction = prepared_item["original_prediction"]
        original_prediction_seconds = prepared_item["original_prediction_seconds"]
        attack_result = prepared_item["attack_result"]
        attack_seconds = prepared_item["attack_seconds"]
        attack_batch_size = prepared_item["attack_batch_size"]

        structured_attack_result = isinstance(
            attack_result,
            (AdversarialAttackResult, AdversarialEvaluationResult),
        )
        if structured_attack_result:
            adversarial_model_input = attack_result.adversarial_model_input
            perturbation_linf = attack_result.perturbation_linf
            perturbation_l2 = attack_result.perturbation_l2
            if isinstance(attack_result, AdversarialAttackResult):
                attack_width, attack_height = attack_result.adversarial_image.size
            else:
                attack_height = int(adversarial_model_input.shape[-2])
                attack_width = int(adversarial_model_input.shape[-1])
        else:
            adversarial_image = attack_result["adversarial_image"]
            adversarial_model_input = model.prepare_attack_input(
                adversarial_image
            ).detach()
            perturbation_linf = attack_result["perturbation_linf"]
            perturbation_l2 = attack_result["perturbation_l2"]
            attack_width, attack_height = adversarial_image.size

        stage_started = start_profile_stage(model_device)

        adversarial_prediction = model.predict_detailed_from_prepared_input(
            prepared_input=adversarial_model_input,
            top_k=5,
        )

        adversarial_prediction_seconds = finish_profile_stage(
            stage_started,
            model_device,
        )
        profile_totals["adversarial_prediction"] += adversarial_prediction_seconds

        if structured_attack_result:
            reconstruction = reconstruct_attack_in_source_geometry(
                source_image=image,
                model_space_delta=attack_result.model_space_delta,
            )
            adversarial_image_to_persist = reconstruction.adversarial_image
            perturbation_image_to_persist = reconstruction.perturbation_visualization
            del reconstruction
        else:
            adversarial_image_to_persist = adversarial_image
            perturbation_image_to_persist = attack_result["perturbation_image"]

        del attack_result

        resolved_label = resolved_by_image_id.get(image_id)

        ground_truth = (
            resolved_label.class_reference if resolved_label is not None else None
        )

        unresolved_label = unresolved_by_image_id.get(image_id)

        label_resolution_reason = None

        if ground_truth is None:
            if unresolved_label is not None:
                label_resolution_reason = unresolved_label.reason
            else:
                label_resolution_reason = "Image has no ground-truth label"

        diagnostics = build_evaluation_image_diagnostics(
            ground_truth=ground_truth,
            original=original_prediction,
            adversarial=adversarial_prediction,
            label_resolution_reason=(label_resolution_reason),
        )

        original_correct = None
        adversarial_correct = None

        if ground_truth is not None:
            original_correct = original_prediction.class_index == ground_truth.index
            adversarial_correct = (
                adversarial_prediction.class_index == ground_truth.index
            )

        prediction_changed = (
            original_prediction.class_index != adversarial_prediction.class_index
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
            adversarial_prediction.confidence - original_prediction.confidence,
            4,
        )

        safe_image_id = image_id.replace("/", "_")
        artifact_filenames = evaluation_image_artifact_filenames(
            result_id,
            image_id,
        )

        stage_started = perf_counter()

        save_result_image(
            image=image,
            result_id=result_id,
            result_path=result_path,
            filename=artifact_filenames["original_visual"],
        )

        adversarial_image_url = save_result_image(
            image=adversarial_image_to_persist,
            result_id=result_id,
            result_path=result_path,
            filename=(f"{safe_image_id}_adversarial.png"),
        )

        perturbation_image_url = save_result_image(
            image=perturbation_image_to_persist,
            result_id=result_id,
            result_path=result_path,
            filename=(f"{safe_image_id}_perturbation.png"),
        )

        save_evaluation_model_inputs(
            result_path=result_path,
            filename=artifact_filenames["model_inputs"],
            evaluation_id=result_id,
            image_id=image_id,
            model_id=model.id,
            clean_model_input=clean_model_input,
            adversarial_model_input=adversarial_model_input,
        )

        evaluation_images_manifest[image_id] = {
            "image_id": image_id,
            "filename": filename,
            "original_visual_filename": (artifact_filenames["original_visual"]),
            "adversarial_visual_filename": (f"{safe_image_id}_adversarial.png"),
            "model_inputs_filename": artifact_filenames["model_inputs"],
            "original_gradcam_filename": (artifact_filenames["original_gradcam"]),
            "adversarial_gradcam_filename": (artifact_filenames["adversarial_gradcam"]),
            "gradcam_generated": False,
        }

        save_images_seconds = perf_counter() - stage_started
        profile_totals["save_images"] += save_images_seconds

        original_image_url = f"/datasets/{request.dataset_id}/images/{image_id}"

        results.append(
            EvaluationImageResult(
                image_id=image_id,
                filename=filename,
                original_image_url=original_image_url,
                adversarial_image_url=(adversarial_image_url),
                perturbation_image_url=(perturbation_image_url),
                ground_truth_label=image_metadata.label,
                original_label=(original_prediction.class_label),
                adversarial_label=(adversarial_prediction.class_label),
                original_confidence=original_confidence,
                adversarial_confidence=(adversarial_confidence),
                confidence_difference=(confidence_difference),
                prediction_changed=prediction_changed,
                # Legacy field. The new UI uses
                # diagnostics.transition and Conditional ASR.
                attack_success=prediction_changed,
                original_correct=original_correct,
                adversarial_correct=adversarial_correct,
                perturbation_linf=perturbation_linf,
                perturbation_l2=perturbation_l2,
                diagnostics=diagnostics,
            )
        )

        metric_samples.append(
            ClassificationEvaluationSample(
                ground_truth=ground_truth,
                original=original_prediction,
                adversarial=adversarial_prediction,
                perturbation_linf=perturbation_linf,
                perturbation_l2=perturbation_l2,
            )
        )

        image_total_seconds = (
            load_image_seconds
            + original_prediction_seconds
            + attack_seconds
            + adversarial_prediction_seconds
            + save_images_seconds
        )

        logger.debug(
            (
                "classification_evaluation_image_profile "
                "image_id=%s filename=%s "
                "source_size=%sx%s attack_size=%sx%s "
                "load_image=%.4fs "
                "original_prediction=%.4fs "
                "attack=%.4fs attack_batch_size=%s "
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
            attack_batch_size,
            adversarial_prediction_seconds,
            save_images_seconds,
            image_total_seconds,
        )

        del adversarial_image_to_persist
        del perturbation_image_to_persist
        if not structured_attack_result:
            del adversarial_image
        del clean_model_input
        del adversarial_model_input
        del image

        report_evaluation_progress(
            progress_callback,
            processed=image_index,
            total=progress_total,
        )
        if progress_callback is not None:
            await asyncio.sleep(0)

    report_evaluation_progress(
        progress_callback,
        processed=progress_total,
        total=progress_total,
        phase="metrics",
    )
    if progress_callback is not None:
        await asyncio.sleep(0)

    metrics_started = perf_counter()
    advanced_metrics = calculate_advanced_classification_metrics(
        samples=metric_samples,
        label_resolution=label_resolution.summary,
        class_labels=model.get_class_labels(),
    )
    metrics_seconds = perf_counter() - metrics_started
    profile.add("metrics", metrics_seconds)

    total_images = len(results)

    average_original_confidence = round(
        sum(result.original_confidence for result in results) / total_images,
        4,
    )

    average_adversarial_confidence = round(
        sum(result.adversarial_confidence for result in results) / total_images,
        4,
    )

    average_confidence_difference = round(
        sum(result.confidence_difference for result in results) / total_images,
        4,
    )

    resolved_results = [
        result for result in results if result.original_correct is not None
    ]

    correct_before_attack = sum(
        result.original_correct is True for result in resolved_results
    )

    correct_after_attack = sum(
        result.adversarial_correct is True for result in resolved_results
    )

    robustness = advanced_metrics.robustness

    summary = EvaluationSummary(
        total_images=total_images,
        changed_predictions=(robustness.changed_predictions),
        unchanged_predictions=(robustness.unchanged_predictions),
        attack_success_rate=(robustness.prediction_flip_rate),
        average_original_confidence=(average_original_confidence),
        average_adversarial_confidence=(average_adversarial_confidence),
        average_confidence_difference=(average_confidence_difference),
        average_perturbation_linf=(robustness.average_perturbation_linf),
        average_perturbation_l2=(robustness.average_perturbation_l2),
        # Legacy summary uses only resolved labels so the
        # denominator remains coherent.
        labeled_images=(label_resolution.summary.resolved_labels),
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
        correct_before_attack=(correct_before_attack if resolved_results else None),
        correct_after_attack=(correct_after_attack if resolved_results else None),
        correct_before_wrong_after=(
            robustness.correct_to_incorrect if resolved_results else None
        ),
    )

    response = EvaluationResponse(
        result_id=result_id,
        dataset_id=request.dataset_id,
        dataset_revision=dataset_revision,
        model_id=model.id,
        model_name=model.name,
        attack_type=request.attack_type,
        parameters=request.parameters,
        summary=summary,
        advanced_metrics=advanced_metrics,
        results=results,
    )

    metadata_started = perf_counter()
    provisional_expires_at = datetime.now(timezone.utc) + timedelta(
        seconds=result_retention_seconds
    )
    save_result_metadata(
        result_path=result_path,
        metadata={
            "result_id": result_id,
            "result_type": "evaluation",
            "status": "completed",
            "owner_id": str(dataset.owner_id),
            "dataset_id": request.dataset_id,
            "dataset_revision": dataset_revision,
            "model_id": model.id,
            "attack_type": request.attack_type,
            "parameters": request.parameters.model_dump(),
            "total_images": total_images,
            "resolved_labels": (label_resolution.summary.resolved_labels),
            "unresolved_labels": (label_resolution.summary.unresolved_labels),
            "metrics_schema_version": (advanced_metrics.metrics_schema_version),
            "images": evaluation_images_manifest,
            "retention_seconds": result_retention_seconds,
            "expires_in_hours": result_retention_seconds // 3600,
            "expires_at": provisional_expires_at.isoformat(),
        },
    )
    expires_at = provisional_expires_at
    if execution is not None:
        expires_at = await complete_evaluation_execution(
            execution=execution,
            response=response,
        )
    save_result_metadata(
        result_path=result_path,
        metadata={
            "result_id": result_id,
            "result_type": "evaluation",
            "status": "completed",
            "owner_id": (
                execution.owner_id if execution is not None else str(dataset.owner_id)
            ),
            "dataset_id": request.dataset_id,
            "dataset_revision": dataset_revision,
            "model_id": model.id,
            "attack_type": request.attack_type,
            "parameters": request.parameters.model_dump(mode="json"),
            "total_images": total_images,
            "resolved_labels": label_resolution.summary.resolved_labels,
            "unresolved_labels": label_resolution.summary.unresolved_labels,
            "metrics_schema_version": advanced_metrics.metrics_schema_version,
            "images": evaluation_images_manifest,
            "retention_seconds": result_retention_seconds,
            "expires_in_hours": result_retention_seconds // 3600,
            "expires_at": expires_at.isoformat(),
        },
    )
    metadata_seconds = perf_counter() - metadata_started
    profile.add("evaluation_persistence", metadata_seconds)

    attack_steps = getattr(
        request.parameters,
        "steps",
        None,
    )

    profile.add("image_load", profile_totals["load_image"])
    profile.add("original_inference", profile_totals["original_prediction"])
    profile.add("attack", profile_totals["attack"])
    profile.add("adversarial_inference", profile_totals["adversarial_prediction"])
    profile.add("artifact_persistence", profile_totals["save_images"])
    log_evaluation_profile(
        logger,
        profile.payload(
            model_id=model.id,
            model_name=model.name,
            total_images=total_images,
            device=model_device,
            extra={"attack_steps": attack_steps},
        ),
    )

    return response

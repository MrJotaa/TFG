import asyncio
import logging
from time import perf_counter

from core.cancellation import CancellationToken
from core.errors import BadRequestError
from core.task_compatibility import (
    DETECTION_TASK_TYPE,
    require_attack_task_type,
    require_model_task_type,
)
from ml.attacks.registry import (
    create_attack_runner,
    get_attack,
    resolve_attack_parameters,
)
from ml.models.detection_inference import (
    DetectionEvaluationInferenceConfiguration,
)
from schemas.detection import to_detection_predictions
from schemas.detection_evaluation import (
    DetectionEvaluationImageResult,
    DetectionEvaluationRequest,
    DetectionEvaluationResponse,
    DetectionEvaluationSummary,
)
from services.dataset_service import (
    dataset_read_snapshot,
    get_dataset,
    get_image_from_dataset,
)
from services.detection_annotation_snapshot import (
    read_detection_annotation_snapshot,
)
from services.detection_evaluation_persistence import (
    new_detection_evaluation_id,
    persist_detection_evaluation,
)
from services.detection_model_execution import (
    run_detection_attack_operation,
    run_detection_model_operation,
)
from services.detection_supervised_evaluation import (
    build_detection_supervised_metrics,
    build_legacy_detection_result,
    prepare_detection_supervised_evaluation,
    requires_detection_supervised_catalog,
)
from services.evaluation_cancellation import (
    register_evaluation_cancellation,
    unregister_evaluation_cancellation,
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
from utils.detection_evaluation_artifacts import (
    remove_detection_evaluation_visual_artifacts,
    save_detection_evaluation_adversarial_image,
)
from utils.detection_metrics import (
    average,
    calculate_detection_comparison,
)
from utils.reproducibility import attack_parameters_for_image
from utils.result_files import RESULTS_TTL_SECONDS

logger = logging.getLogger(__name__)


async def evaluate_detection_dataset(
    request: DetectionEvaluationRequest,
    *,
    owner_id: str | None = None,
    retention_seconds: int = RESULTS_TTL_SECONDS,
    progress_callback: EvaluationProgressCallback | None = None,
    cancellation_token: CancellationToken | None = None,
) -> DetectionEvaluationResponse:
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
            task_type="detection",
            retention_seconds=retention_seconds,
        )
    result_id = (
        execution.result_id if execution is not None else new_detection_evaluation_id()
    )
    if execution is not None and cancellation_token is None:
        cancellation_token = CancellationToken()
    if execution is not None:
        register_evaluation_cancellation(execution.result_id, cancellation_token)

    completed = False
    try:
        with dataset_read_snapshot(request.dataset_id):
            response = await _evaluate_detection_dataset(
                request,
                attack=attack,
                result_id=result_id,
                execution=execution,
                progress_callback=progress_callback,
                cancellation_token=cancellation_token,
            )
        completed = True
        return response
    finally:
        try:
            if not completed:
                if execution is not None:
                    await abort_evaluation_execution(execution)
                else:
                    try:
                        remove_detection_evaluation_visual_artifacts(result_id)
                    except OSError:
                        logger.exception(
                            "detection_evaluation_visual_cleanup_failed result_id=%s",
                            result_id,
                        )
        finally:
            if execution is not None:
                unregister_evaluation_cancellation(
                    execution.result_id,
                    cancellation_token,
                )


def _raise_if_cancelled(cancellation_token: CancellationToken | None) -> None:
    if cancellation_token is not None:
        cancellation_token.raise_if_cancelled()


async def _yield_progress(callback: EvaluationProgressCallback | None) -> None:
    if callback is not None:
        await asyncio.sleep(0)


async def _detection_results_for_image(
    *,
    model,
    image,
    adversarial_image,
    supervised_context,
    inference_configuration,
    image_id: str,
    original_supervised_results: dict,
    adversarial_supervised_results: dict,
    model_device,
):
    if supervised_context.available:
        original_scientific_result, original_seconds = await _timed_detection_operation(
            model.detect_for_evaluation,
            image,
            inference_configuration,
            model_device=model_device,
        )
        adversarial_scientific_result, adversarial_seconds = (
            await _timed_detection_operation(
                model.detect_for_evaluation,
                adversarial_image,
                inference_configuration,
                model_device=model_device,
            )
        )
        if image_id in supervised_context.snapshot.reviewed_image_ids:
            original_supervised_results[image_id] = original_scientific_result
            adversarial_supervised_results[image_id] = adversarial_scientific_result
        return (
            build_legacy_detection_result(original_scientific_result),
            build_legacy_detection_result(adversarial_scientific_result),
            original_seconds,
            adversarial_seconds,
        )

    original_result, original_seconds = await _timed_detection_operation(
        model.detect,
        image,
        model_device=model_device,
    )
    adversarial_result, adversarial_seconds = await _timed_detection_operation(
        model.detect,
        adversarial_image,
        model_device=model_device,
    )
    return (
        original_result,
        adversarial_result,
        original_seconds,
        adversarial_seconds,
    )


async def _timed_detection_operation(
    operation,
    *args,
    model_device,
):
    started = start_profile_stage(model_device)
    result = await run_detection_model_operation(operation, *args)
    return result, finish_profile_stage(started, model_device)


def _attack_outcome_counts(metrics) -> tuple[int, int, int]:
    return (
        int(metrics.adversarial_count < metrics.original_count),
        int(metrics.adversarial_count == 0 and metrics.original_count > 0),
        int(metrics.attack_success),
    )


def _safe_rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


async def _evaluate_detection_dataset(
    request: DetectionEvaluationRequest,
    *,
    attack,
    result_id: str,
    execution: EvaluationExecution | None,
    progress_callback: EvaluationProgressCallback | None,
    cancellation_token: CancellationToken | None,
) -> DetectionEvaluationResponse:
    profile = EvaluationProfile(
        task_type="detection",
        dataset_id=request.dataset_id,
        attack_type=request.attack_type.value,
    )

    dataset_started = perf_counter()
    dataset = await get_dataset(request.dataset_id)
    profile.add("dataset_load", perf_counter() - dataset_started)

    setup_started = perf_counter()
    model = await run_detection_model_operation(get_active_model_instance)

    require_model_task_type(
        model=model,
        expected_task_type=DETECTION_TASK_TYPE,
        endpoint="/detect/evaluate",
    )

    require_attack_task_type(
        attack=attack,
        task_type=model.task_type,
    )

    attack_runner = create_attack_runner(
        attack=attack,
        model_wrapper=model,
        parameters=request.parameters,
        cancellation_token=cancellation_token,
    )

    annotation_snapshot = await read_detection_annotation_snapshot(request.dataset_id)
    catalog = (
        model.get_detection_class_catalog()
        if requires_detection_supervised_catalog(annotation_snapshot)
        else None
    )
    supervised_context = prepare_detection_supervised_evaluation(
        snapshot=annotation_snapshot,
        catalog=catalog,
    )
    inference_configuration = (
        DetectionEvaluationInferenceConfiguration(
            score_floor=request.supervised_configuration.score_floor,
        )
        if supervised_context.available
        else None
    )
    profile.add("setup", perf_counter() - setup_started)
    model_device = get_model_device(model)

    profile_totals = {
        "image_load": 0.0,
        "attack": 0.0,
        "artifact_persistence": 0.0,
        "original_inference": 0.0,
        "adversarial_inference": 0.0,
        "per_image_metrics": 0.0,
    }

    image_results = []
    original_supervised_results = {}
    adversarial_supervised_results = {}

    total_original_detections = 0
    total_adversarial_detections = 0
    images_with_reduced_detections = 0
    images_with_no_adversarial_detections = 0
    successful_attacks = 0

    original_confidences = []
    adversarial_confidences = []
    confidence_differences = []
    perturbation_linf_values = []
    perturbation_l2_values = []
    progress_total = len(dataset.images)
    report_evaluation_progress(
        progress_callback,
        processed=0,
        total=progress_total,
    )

    for image_index, image_metadata in enumerate(dataset.images, start=1):
        image_started = perf_counter()

        stage_started = perf_counter()
        image = await get_image_from_dataset(
            dataset_id=request.dataset_id,
            image_id=image_metadata.id,
        )
        image_load_seconds = perf_counter() - stage_started
        profile_totals["image_load"] += image_load_seconds

        stage_started = start_profile_stage(model_device)
        _raise_if_cancelled(cancellation_token)
        attack_result = await run_detection_attack_operation(
            attack_runner,
            model_wrapper=model,
            image=image,
            parameters=attack_parameters_for_image(
                attack_type=request.attack_type,
                parameters=request.parameters,
                image_id=image_metadata.id,
            ),
        )
        _raise_if_cancelled(cancellation_token)
        attack_seconds = finish_profile_stage(stage_started, model_device)
        profile_totals["attack"] += attack_seconds

        adversarial_image = attack_result["adversarial_image"]
        stage_started = perf_counter()
        adversarial_image_url = await asyncio.to_thread(
            save_detection_evaluation_adversarial_image,
            evaluation_id=result_id,
            image_id=image_metadata.id,
            image=adversarial_image,
        )
        artifact_persistence_seconds = perf_counter() - stage_started
        profile_totals["artifact_persistence"] += artifact_persistence_seconds

        (
            original_detection_result,
            adversarial_detection_result,
            original_inference_seconds,
            adversarial_inference_seconds,
        ) = await _detection_results_for_image(
            model=model,
            image=image,
            adversarial_image=adversarial_image,
            supervised_context=supervised_context,
            inference_configuration=inference_configuration,
            image_id=image_metadata.id,
            original_supervised_results=original_supervised_results,
            adversarial_supervised_results=adversarial_supervised_results,
            model_device=model_device,
        )

        profile_totals["original_inference"] += original_inference_seconds
        profile_totals["adversarial_inference"] += adversarial_inference_seconds

        stage_started = perf_counter()
        original_detections = original_detection_result.detections
        adversarial_detections = adversarial_detection_result.detections

        metrics = calculate_detection_comparison(
            original_detections=original_detections,
            adversarial_detections=adversarial_detections,
        )

        reduced_count, empty_count, successful_count = _attack_outcome_counts(metrics)
        images_with_reduced_detections += reduced_count
        images_with_no_adversarial_detections += empty_count
        successful_attacks += successful_count

        total_original_detections += metrics.original_count
        total_adversarial_detections += metrics.adversarial_count

        original_confidences.append(metrics.average_original_confidence)
        adversarial_confidences.append(metrics.average_adversarial_confidence)
        confidence_differences.append(metrics.average_confidence_difference)
        perturbation_linf_values.append(attack_result["perturbation_linf"])
        perturbation_l2_values.append(attack_result["perturbation_l2"])

        image_results.append(
            DetectionEvaluationImageResult(
                image_id=image_metadata.id,
                filename=image_metadata.filename,
                original_detection_count=metrics.original_count,
                adversarial_detection_count=metrics.adversarial_count,
                detections_difference=metrics.detections_difference,
                average_original_confidence=round(
                    metrics.average_original_confidence,
                    4,
                ),
                average_adversarial_confidence=round(
                    metrics.average_adversarial_confidence,
                    4,
                ),
                average_confidence_difference=round(
                    metrics.average_confidence_difference,
                    4,
                ),
                attack_success=metrics.attack_success,
                perturbation_linf=attack_result["perturbation_linf"],
                perturbation_l2=attack_result["perturbation_l2"],
                adversarial_image_url=adversarial_image_url,
                original_detections=to_detection_predictions(original_detections),
                adversarial_detections=to_detection_predictions(adversarial_detections),
            )
        )
        per_image_metrics_seconds = perf_counter() - stage_started
        profile_totals["per_image_metrics"] += per_image_metrics_seconds

        logger.debug(
            (
                "detection_evaluation_image_profile "
                "image_id=%s filename=%s "
                "image_load=%.4fs attack=%.4fs "
                "artifact_persistence=%.4fs "
                "original_inference=%.4fs adversarial_inference=%.4fs "
                "per_image_metrics=%.4fs total=%.4fs"
            ),
            image_metadata.id,
            image_metadata.filename,
            image_load_seconds,
            attack_seconds,
            artifact_persistence_seconds,
            original_inference_seconds,
            adversarial_inference_seconds,
            per_image_metrics_seconds,
            perf_counter() - image_started,
        )

        del adversarial_image
        del image
        del attack_result

        report_evaluation_progress(
            progress_callback,
            processed=image_index,
            total=progress_total,
        )
        await _yield_progress(progress_callback)

    report_evaluation_progress(
        progress_callback,
        processed=progress_total,
        total=progress_total,
        phase="metrics",
    )
    await _yield_progress(progress_callback)

    summary_started = perf_counter()
    total_images = len(image_results)

    summary = DetectionEvaluationSummary(
        total_images=total_images,
        total_original_detections=total_original_detections,
        total_adversarial_detections=total_adversarial_detections,
        detections_difference=(
            total_original_detections - total_adversarial_detections
        ),
        average_original_detections_per_image=_safe_rate(
            total_original_detections,
            total_images,
        ),
        average_adversarial_detections_per_image=_safe_rate(
            total_adversarial_detections,
            total_images,
        ),
        average_original_confidence=average(original_confidences),
        average_adversarial_confidence=average(adversarial_confidences),
        average_confidence_difference=average(confidence_differences),
        images_with_reduced_detections=images_with_reduced_detections,
        images_with_no_adversarial_detections=images_with_no_adversarial_detections,
        attack_success_rate=_safe_rate(successful_attacks, total_images),
        average_perturbation_linf=average(perturbation_linf_values),
        average_perturbation_l2=average(perturbation_l2_values),
    )
    profile.add("summary_metrics", perf_counter() - summary_started)

    supervised_metrics_started = perf_counter()
    supervised = await asyncio.to_thread(
        build_detection_supervised_metrics,
        context=supervised_context,
        original_results_by_image_id=original_supervised_results,
        adversarial_results_by_image_id=adversarial_supervised_results,
        configuration=request.supervised_configuration,
    )
    profile.add(
        "supervised_metrics",
        perf_counter() - supervised_metrics_started,
    )

    response = DetectionEvaluationResponse(
        result_id=result_id,
        model_id=model.id,
        model_name=model.name,
        dataset_id=request.dataset_id,
        dataset_revision=annotation_snapshot.dataset_revision,
        annotation_snapshot_id=annotation_snapshot.snapshot_id,
        attack_type=request.attack_type,
        parameters=request.parameters,
        summary=summary,
        results=image_results,
        supervised=supervised,
    )

    persistence_started = perf_counter()
    if execution is None:
        await persist_detection_evaluation(
            response=response,
            request=request,
            owner_id=dataset.owner_id,
        )
    else:
        cancellation_token.raise_if_cancelled()
        await complete_evaluation_execution(
            execution=execution,
            response=response,
        )
    profile.add("evaluation_persistence", perf_counter() - persistence_started)

    for stage, seconds in profile_totals.items():
        profile.add(stage, seconds)

    log_evaluation_profile(
        logger,
        profile.payload(
            model_id=model.id,
            model_name=model.name,
            total_images=total_images,
            device=model_device,
            extra={
                "supervised_metrics_available": supervised_context.available,
                "reviewed_images": len(supervised_context.snapshot.reviewed_image_ids),
            },
        ),
    )
    return response

import logging

from core.errors import BadRequestError
from core.task_compatibility import (
    DETECTION_TASK_TYPE,
    require_attack_task_type,
    require_model_task_type,
)
from ml.attacks.registry import get_attack
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
from services.dataset_service import get_dataset, get_image_from_dataset
from services.detection_annotation_snapshot import (
    read_detection_annotation_snapshot,
)
from services.detection_evaluation_persistence import (
    new_detection_evaluation_id,
    persist_detection_evaluation,
)
from services.detection_supervised_evaluation import (
    build_detection_supervised_metrics,
    build_legacy_detection_result,
    prepare_detection_supervised_evaluation,
    requires_detection_supervised_catalog,
)
from services.evaluation_history_service import (
    EvaluationExecution,
    abort_evaluation_execution,
    begin_evaluation_execution,
    complete_evaluation_execution,
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
) -> DetectionEvaluationResponse:
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
    completed = False
    try:
        response = await _evaluate_detection_dataset(
            request,
            result_id=result_id,
            execution=execution,
        )
        completed = True
        return response
    finally:
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


async def _evaluate_detection_dataset(
    request: DetectionEvaluationRequest,
    *,
    result_id: str,
    execution: EvaluationExecution | None,
) -> DetectionEvaluationResponse:
    dataset = await get_dataset(request.dataset_id)

    model = get_active_model_instance()

    require_model_task_type(
        model=model,
        expected_task_type=DETECTION_TASK_TYPE,
        endpoint="/detect/evaluate",
    )

    try:
        attack = get_attack(request.attack_type)
    except ValueError as error:
        raise BadRequestError(str(error)) from error

    require_attack_task_type(
        attack=attack,
        task_type=model.task_type,
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

    for image_metadata in dataset.images:
        image = await get_image_from_dataset(
            dataset_id=request.dataset_id,
            image_id=image_metadata.id,
        )

        attack_result = attack.apply(
            model_wrapper=model,
            image=image,
            parameters=attack_parameters_for_image(
                attack_type=request.attack_type,
                parameters=request.parameters,
                image_id=image_metadata.id,
            ),
        )

        adversarial_image = attack_result["adversarial_image"]
        adversarial_image_url = save_detection_evaluation_adversarial_image(
            evaluation_id=result_id,
            image_id=image_metadata.id,
            image=adversarial_image,
        )

        if supervised_context.available:
            original_scientific_result = model.detect_for_evaluation(
                image,
                inference_configuration,
            )
            adversarial_scientific_result = model.detect_for_evaluation(
                adversarial_image,
                inference_configuration,
            )
            if image_metadata.id in supervised_context.snapshot.reviewed_image_ids:
                original_supervised_results[image_metadata.id] = (
                    original_scientific_result
                )
                adversarial_supervised_results[image_metadata.id] = (
                    adversarial_scientific_result
                )

            original_detection_result = build_legacy_detection_result(
                original_scientific_result
            )
            adversarial_detection_result = build_legacy_detection_result(
                adversarial_scientific_result
            )
        else:
            original_detection_result = model.detect(image)
            adversarial_detection_result = model.detect(adversarial_image)

        original_detections = original_detection_result.detections
        adversarial_detections = adversarial_detection_result.detections

        metrics = calculate_detection_comparison(
            original_detections=original_detections,
            adversarial_detections=adversarial_detections,
        )

        if metrics.adversarial_count < metrics.original_count:
            images_with_reduced_detections += 1

        if metrics.adversarial_count == 0 and metrics.original_count > 0:
            images_with_no_adversarial_detections += 1

        if metrics.attack_success:
            successful_attacks += 1

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
                adversarial_detection_count=(metrics.adversarial_count),
                detections_difference=(metrics.detections_difference),
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

    total_images = len(image_results)

    summary = DetectionEvaluationSummary(
        total_images=total_images,
        total_original_detections=total_original_detections,
        total_adversarial_detections=total_adversarial_detections,
        detections_difference=(
            total_original_detections - total_adversarial_detections
        ),
        average_original_detections_per_image=round(
            total_original_detections / total_images,
            4,
        )
        if total_images
        else 0.0,
        average_adversarial_detections_per_image=round(
            total_adversarial_detections / total_images,
            4,
        )
        if total_images
        else 0.0,
        average_original_confidence=average(original_confidences),
        average_adversarial_confidence=average(adversarial_confidences),
        average_confidence_difference=average(confidence_differences),
        images_with_reduced_detections=images_with_reduced_detections,
        images_with_no_adversarial_detections=images_with_no_adversarial_detections,
        attack_success_rate=round(successful_attacks / total_images, 4)
        if total_images
        else 0.0,
        average_perturbation_linf=average(perturbation_linf_values),
        average_perturbation_l2=average(perturbation_l2_values),
    )
    supervised = build_detection_supervised_metrics(
        context=supervised_context,
        original_results_by_image_id=original_supervised_results,
        adversarial_results_by_image_id=adversarial_supervised_results,
        configuration=request.supervised_configuration,
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
    if execution is None:
        await persist_detection_evaluation(
            response=response,
            request=request,
            owner_id=dataset.owner_id,
        )
    else:
        await complete_evaluation_execution(
            execution=execution,
            response=response,
        )
    return response

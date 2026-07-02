from ml.attacks.registry import get_attack
from schemas.detection_evaluation import (
    DetectionEvaluationImageResult,
    DetectionEvaluationRequest,
    DetectionEvaluationResponse,
    DetectionEvaluationSummary,
)
from services.dataset_service import get_dataset, get_image_from_dataset
from services.model_service import get_active_model_instance
from core.task_compatibility import (
    DETECTION_TASK_TYPE,
    require_attack_task_type,
    require_model_task_type,
)
from core.errors import BadRequestError
from utils.detection_metrics import (
    average,
    calculate_detection_comparison,
)


async def evaluate_detection_dataset(
    request: DetectionEvaluationRequest,
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

    image_results = []

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
            parameters=request.parameters,
        )

        adversarial_image = attack_result["adversarial_image"]

        original_detection_response = model.detect(image)
        original_detections = original_detection_response.detections

        adversarial_detection_response = model.detect(adversarial_image)
        adversarial_detections = adversarial_detection_response.detections

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
                average_original_confidence=(metrics.average_original_confidence),
                average_adversarial_confidence=(metrics.average_adversarial_confidence),
                average_confidence_difference=(metrics.average_confidence_difference),
                attack_success=metrics.attack_success,
                perturbation_linf=attack_result["perturbation_linf"],
                perturbation_l2=attack_result["perturbation_l2"],
                original_detections=original_detections,
                adversarial_detections=adversarial_detections,
            )
        )

    total_images = len(image_results)

    summary = DetectionEvaluationSummary(
        total_images=total_images,
        total_original_detections=total_original_detections,
        total_adversarial_detections=total_adversarial_detections,
        detections_difference=(
            total_adversarial_detections - total_original_detections
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

    return DetectionEvaluationResponse(
        model_id=model.id,
        model_name=model.name,
        dataset_id=request.dataset_id,
        attack_type=request.attack_type,
        parameters=request.parameters,
        summary=summary,
        results=image_results,
    )

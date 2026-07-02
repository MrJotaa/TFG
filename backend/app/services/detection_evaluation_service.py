from fastapi import HTTPException

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


def _average_detection_confidence(detections) -> float:
    if not detections:
        return 0.0

    total_confidence = sum(detection.confidence for detection in detections)

    return round(total_confidence / len(detections), 4)


def _average(values: list[float]) -> float:
    if not values:
        return 0.0

    return round(sum(values) / len(values), 4)


async def evaluate_detection_dataset(
    request: DetectionEvaluationRequest,
) -> DetectionEvaluationResponse:
    try:
        dataset = await get_dataset(request.dataset_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    try:
        model = get_active_model_instance()
    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    require_model_task_type(
        model=model,
        expected_task_type=DETECTION_TASK_TYPE,
        endpoint="/detect/evaluate",
    )

    try:
        attack = get_attack(request.attack_type)
    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

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

        adversarial_detection_response = model.detect(
            adversarial_image
        )
        adversarial_detections = (
            adversarial_detection_response.detections
        )

        original_count = len(original_detections)
        adversarial_count = len(adversarial_detections)
        detections_difference = adversarial_count - original_count

        average_original_confidence = (
            _average_detection_confidence(
                original_detections
            )
        )
        average_adversarial_confidence = (
            _average_detection_confidence(
                adversarial_detections
            )
        )
        average_confidence_difference = round(
            average_adversarial_confidence
            - average_original_confidence,
            4,
        )

        attack_success = (
            adversarial_count < original_count
            or average_confidence_difference < 0
        )

        if adversarial_count < original_count:
            images_with_reduced_detections += 1

        if adversarial_count == 0 and original_count > 0:
            images_with_no_adversarial_detections += 1

        if attack_success:
            successful_attacks += 1

        total_original_detections += original_count
        total_adversarial_detections += adversarial_count

        original_confidences.append(
            average_original_confidence
        )
        adversarial_confidences.append(
            average_adversarial_confidence
        )
        confidence_differences.append(
            average_confidence_difference
        )
        perturbation_linf_values.append(
            attack_result["perturbation_linf"]
        )
        perturbation_l2_values.append(
            attack_result["perturbation_l2"]
        )

        image_results.append(
            DetectionEvaluationImageResult(
                image_id=image_metadata.id,
                filename=image_metadata.filename,
                original_detection_count=original_count,
                adversarial_detection_count=adversarial_count,
                detections_difference=detections_difference,
                average_original_confidence=(
                    average_original_confidence
                ),
                average_adversarial_confidence=(
                    average_adversarial_confidence
                ),
                average_confidence_difference=(
                    average_confidence_difference
                ),
                attack_success=attack_success,
                perturbation_linf=attack_result[
                    "perturbation_linf"
                ],
                perturbation_l2=attack_result[
                    "perturbation_l2"
                ],
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
        average_original_confidence=_average(original_confidences),
        average_adversarial_confidence=_average(adversarial_confidences),
        average_confidence_difference=_average(confidence_differences),
        images_with_reduced_detections=images_with_reduced_detections,
        images_with_no_adversarial_detections=images_with_no_adversarial_detections,
        attack_success_rate=round(successful_attacks / total_images, 4)
        if total_images
        else 0.0,
        average_perturbation_linf=_average(perturbation_linf_values),
        average_perturbation_l2=_average(perturbation_l2_values),
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
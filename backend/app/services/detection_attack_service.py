from fastapi import HTTPException

from ml.attacks.registry import get_attack
from schemas.detection_attack import (
    DetectionAttackComparison,
    DetectionAttackRequest,
    DetectionAttackResponse,
)
from services.dataset_service import get_image_from_dataset
from services.model_service import get_active_model_instance
from utils.result_files import (
    create_result_directory,
    save_result_image,
    save_result_metadata,
)
from ml.explainability.gradcam import (
    draw_detections_on_image,
    generate_detection_eigencam,
)
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


async def apply_detection_attack(
    request: DetectionAttackRequest,
) -> DetectionAttackResponse:
    try:
        image = await get_image_from_dataset(
            dataset_id=request.dataset_id,
            image_id=request.image_id,
        )
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
        endpoint="/detect/attack",
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

    original_detection_response, original_annotated_image = (
        model.detect_with_visualization(image)
    )

    attack_result = attack.apply(
        model_wrapper=model,
        image=image,
        parameters=request.parameters,
    )

    adversarial_image = attack_result["adversarial_image"]
    perturbation_image = attack_result["perturbation_image"]

    adversarial_detection_response, adversarial_annotated_image = (
        model.detect_with_visualization(adversarial_image)
    )

    original_detections = original_detection_response.detections
    adversarial_detections = adversarial_detection_response.detections

    original_explanation_image = draw_detections_on_image(
        image=generate_detection_eigencam(model, image),
        detections=original_detections,
    )

    adversarial_explanation_image = draw_detections_on_image(
        image=generate_detection_eigencam(model, adversarial_image),
        detections=adversarial_detections,
    )

    original_count = len(original_detections)
    adversarial_count = len(adversarial_detections)
    detections_difference = adversarial_count - original_count

    average_original_confidence = _average_detection_confidence(original_detections)
    average_adversarial_confidence = _average_detection_confidence(
        adversarial_detections
    )
    average_confidence_difference = round(
        average_adversarial_confidence - average_original_confidence,
        4,
    )

    attack_success = (
        adversarial_count < original_count or average_confidence_difference < 0
    )

    comparison = DetectionAttackComparison(
        original_detection_count=original_count,
        adversarial_detection_count=adversarial_count,
        detections_difference=detections_difference,
        average_original_confidence=average_original_confidence,
        average_adversarial_confidence=average_adversarial_confidence,
        average_confidence_difference=average_confidence_difference,
        attack_success=attack_success,
        perturbation_linf=attack_result["perturbation_linf"],
        perturbation_l2=attack_result["perturbation_l2"],
    )

    result_id, result_path = create_result_directory(prefix="detection_attack")

    original_annotated_image_url = save_result_image(
        image=original_annotated_image,
        result_id=result_id,
        result_path=result_path,
        filename="original_annotated.png",
    )

    adversarial_image_url = save_result_image(
        image=adversarial_image,
        result_id=result_id,
        result_path=result_path,
        filename="adversarial.png",
    )

    adversarial_annotated_image_url = save_result_image(
        image=adversarial_annotated_image,
        result_id=result_id,
        result_path=result_path,
        filename="adversarial_annotated.png",
    )

    perturbation_image_url = save_result_image(
        image=perturbation_image,
        result_id=result_id,
        result_path=result_path,
        filename="perturbation.png",
    )

    original_explanation_image_url = save_result_image(
        image=original_explanation_image,
        result_id=result_id,
        result_path=result_path,
        filename="original_explanation.png",
    )

    adversarial_explanation_image_url = save_result_image(
        image=adversarial_explanation_image,
        result_id=result_id,
        result_path=result_path,
        filename="adversarial_explanation.png",
    )

    save_result_metadata(
        result_path=result_path,
        metadata={
            "result_id": result_id,
            "result_type": "detection_attack",
            "dataset_id": request.dataset_id,
            "image_id": request.image_id,
            "model_id": model.id,
            "model_name": model.name,
            "attack_type": request.attack_type,
            "parameters": request.parameters.model_dump(),
            "original_detection_count": original_count,
            "adversarial_detection_count": adversarial_count,
            "explanation_type": "eigencam",
            "original_explanation_image": "original_explanation.png",
            "adversarial_explanation_image": "adversarial_explanation.png",
            "expires_in_hours": 24,
        },
    )

    return DetectionAttackResponse(
        result_id=result_id,
        attack_type=request.attack_type,
        parameters=request.parameters,
        original_annotated_image_url=original_annotated_image_url,
        adversarial_image_url=adversarial_image_url,
        adversarial_annotated_image_url=adversarial_annotated_image_url,
        perturbation_image_url=perturbation_image_url,
        original_detections=original_detections,
        adversarial_detections=adversarial_detections,
        comparison=comparison,
        original_explanation_image_url=original_explanation_image_url,
        adversarial_explanation_image_url=adversarial_explanation_image_url,
    )

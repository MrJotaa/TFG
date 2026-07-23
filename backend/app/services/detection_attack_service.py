from ml.attacks.registry import get_attack
from schemas.detection_attack import (
    DetectionAttackComparison,
    DetectionAttackRequest,
    DetectionAttackResponse,
)
from schemas.detection import to_detection_predictions
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
from core.errors import BadRequestError
from utils.detection_metrics import (
    calculate_detection_comparison,
)


async def apply_detection_attack(
    request: DetectionAttackRequest,
) -> DetectionAttackResponse:
    image = await get_image_from_dataset(
        dataset_id=request.dataset_id,
        image_id=request.image_id,
    )

    model = get_active_model_instance()

    require_model_task_type(
        model=model,
        expected_task_type=DETECTION_TASK_TYPE,
        endpoint="/detect/attack",
    )

    try:
        attack = get_attack(request.attack_type)
    except ValueError as error:
        raise BadRequestError(str(error)) from error

    require_attack_task_type(
        attack=attack,
        task_type=model.task_type,
    )

    original_detection_result, original_annotated_image = (
        model.detect_with_visualization(image)
    )

    attack_result = attack.apply(
        model_wrapper=model,
        image=image,
        parameters=request.parameters,
    )

    adversarial_image = attack_result["adversarial_image"]
    perturbation_image = attack_result["perturbation_image"]

    adversarial_detection_result, adversarial_annotated_image = (
        model.detect_with_visualization(adversarial_image)
    )

    original_detections = original_detection_result.detections
    adversarial_detections = adversarial_detection_result.detections

    original_explanation_image = draw_detections_on_image(
        image=generate_detection_eigencam(model, image),
        detections=original_detections,
    )

    adversarial_explanation_image = draw_detections_on_image(
        image=generate_detection_eigencam(model, adversarial_image),
        detections=adversarial_detections,
    )

    metrics = calculate_detection_comparison(
        original_detections=original_detections,
        adversarial_detections=adversarial_detections,
    )

    comparison = DetectionAttackComparison(
        original_detection_count=metrics.original_count,
        adversarial_detection_count=metrics.adversarial_count,
        detections_difference=metrics.detections_difference,
        average_original_confidence=round(metrics.average_original_confidence, 4),
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
            "original_detection_count": metrics.original_count,
            "adversarial_detection_count": metrics.adversarial_count,
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
        original_detections=to_detection_predictions(original_detections),
        adversarial_detections=to_detection_predictions(adversarial_detections),
        comparison=comparison,
        original_explanation_image_url=original_explanation_image_url,
        adversarial_explanation_image_url=adversarial_explanation_image_url,
    )

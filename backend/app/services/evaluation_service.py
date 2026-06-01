from fastapi import HTTPException

from bson import ObjectId

from db.collections import datasets_collection
from ml.attacks.registry import get_attack
from schemas.evaluation import (
    EvaluationImageResult,
    EvaluationRequest,
    EvaluationResponse,
    EvaluationSummary,
)
from services.dataset_service import get_image_from_dataset
from services.model_service import get_active_model_instance
from utils.result_files import (
    create_result_directory,
    save_result_image,
    save_result_metadata,
)


async def evaluate_dataset(request: EvaluationRequest) -> EvaluationResponse:
    if not ObjectId.is_valid(request.dataset_id):
        raise HTTPException(status_code=400, detail="Invalid dataset id")

    dataset = await datasets_collection.find_one({"_id": ObjectId(request.dataset_id)})

    if not dataset:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset with id {request.dataset_id} not found",
        )

    images_metadata = dataset.get("images", [])

    if not images_metadata:
        raise HTTPException(
            status_code=400,
            detail="Dataset must contain at least one image",
        )

    try:
        model = get_active_model_instance()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        attack = get_attack(request.attack_type)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    result_id, result_path = create_result_directory(prefix="evaluation")

    results: list[EvaluationImageResult] = []

    for image_metadata in images_metadata:
        image_id = image_metadata["id"]
        filename = image_metadata["filename"]

        image = await get_image_from_dataset(
            dataset_id=request.dataset_id,
            image_id=image_id,
        )

        original_prediction = model.predict(image)

        try:
            attack_result = attack.apply(
                model_wrapper=model,
                image=image,
                parameters=request.parameters,
            )
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Error attacking image {filename}: {str(e)}",
            )

        adversarial_image = attack_result["adversarial_image"]
        perturbation_image = attack_result["perturbation_image"]

        adversarial_prediction = model.predict(adversarial_image)

        confidence_difference = round(
            adversarial_prediction.confidence - original_prediction.confidence,
            4,
        )

        prediction_changed = (
            original_prediction.prediction != adversarial_prediction.prediction
        )

        safe_image_id = image_id.replace("/", "_")

        adversarial_image_url = save_result_image(
            image=adversarial_image,
            result_id=result_id,
            result_path=result_path,
            filename=f"{safe_image_id}_adversarial.png",
        )

        perturbation_image_url = save_result_image(
            image=perturbation_image,
            result_id=result_id,
            result_path=result_path,
            filename=f"{safe_image_id}_perturbation.png",
        )

        original_image_url = f"/datasets/{request.dataset_id}/images/{image_id}"

        results.append(
            EvaluationImageResult(
                image_id=image_id,
                filename=filename,
                original_image_url=original_image_url,
                adversarial_image_url=adversarial_image_url,
                perturbation_image_url=perturbation_image_url,
                original_label=original_prediction.prediction,
                adversarial_label=adversarial_prediction.prediction,
                original_confidence=original_prediction.confidence,
                adversarial_confidence=adversarial_prediction.confidence,
                confidence_difference=confidence_difference,
                prediction_changed=prediction_changed,
                attack_success=prediction_changed,
                perturbation_linf=attack_result["perturbation_linf"],
                perturbation_l2=attack_result["perturbation_l2"],
            )
        )

    total_images = len(results)
    changed_predictions = sum(1 for result in results if result.prediction_changed)
    unchanged_predictions = total_images - changed_predictions

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

    average_perturbation_linf = round(
        sum(result.perturbation_linf for result in results) / total_images,
        4,
    )

    average_perturbation_l2 = round(
        sum(result.perturbation_l2 for result in results) / total_images,
        4,
    )

    attack_success_rate = round(changed_predictions / total_images, 4)

    summary = EvaluationSummary(
        total_images=total_images,
        changed_predictions=changed_predictions,
        unchanged_predictions=unchanged_predictions,
        attack_success_rate=attack_success_rate,
        average_original_confidence=average_original_confidence,
        average_adversarial_confidence=average_adversarial_confidence,
        average_confidence_difference=average_confidence_difference,
        average_perturbation_linf=average_perturbation_linf,
        average_perturbation_l2=average_perturbation_l2,
    )

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
            "expires_in_hours": 24,
        },
    )

    return EvaluationResponse(
        result_id=result_id,
        dataset_id=request.dataset_id,
        model_id=model.id,
        attack_type=request.attack_type,
        parameters=request.parameters,
        summary=summary,
        results=results,
    )
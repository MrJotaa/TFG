import asyncio

from PIL import Image, ImageDraw

from core.errors import BadRequestError, GoneError, NotFoundError
from core.task_compatibility import (
    DETECTION_TASK_TYPE,
    require_model_task_type,
)
from ml.explainability.gradcam import generate_detection_eigencam
from schemas.detection import DetectionPrediction
from schemas.detection_evaluation import (
    DetectionEvaluationEigenCamResponse,
    DetectionEvaluationImageResult,
)
from services.dataset_service import get_image_from_dataset
from services.detection_model_execution import run_detection_model_operation
from services.evaluation_history_service import get_stored_evaluation
from services.model_service import get_active_model_instance
from utils.detection_evaluation_artifacts import (
    detection_eigencam_pair_is_available,
    detection_evaluation_image_artifact_filenames,
    load_detection_evaluation_adversarial_image,
    save_detection_eigencam_pair,
)

_generation_lock = asyncio.Lock()


async def generate_detection_evaluation_image_eigencam(
    *,
    evaluation_id: str,
    image_id: str,
    current_user_id: str,
) -> DetectionEvaluationEigenCamResponse:
    stored = await get_stored_evaluation(
        evaluation_id=evaluation_id,
        owner_id=current_user_id,
    )
    if stored.task_type != "detection":
        raise BadRequestError("La evaluación solicitada no es de detección")
    evaluation = stored.evaluation
    image_result = _resolve_image_result(evaluation.results, image_id)

    async with _generation_lock:
        if detection_eigencam_pair_is_available(
            evaluation_id=evaluation_id,
            image_id=image_id,
        ):
            return _build_response(evaluation_id, image_id)

        if not image_result.adversarial_image_url:
            raise GoneError(
                "Los datos necesarios para la explicabilidad de la evaluación de detección ya no están disponibles; "
                "repite la evaluación"
            )

        model = await run_detection_model_operation(get_active_model_instance)
        require_model_task_type(
            model=model,
            expected_task_type=DETECTION_TASK_TYPE,
            endpoint=(
                f"/detect/evaluations/{evaluation_id}/images/{image_id}/eigencam"
            ),
        )
        if model.id != evaluation.model_id:
            raise BadRequestError(
                "El modelo utilizado en esta evaluación de detección no está activo"
            )

        original_image = await get_image_from_dataset(
            dataset_id=evaluation.dataset_id,
            image_id=image_id,
        )
        original_image = original_image.copy().convert("RGB")
        adversarial_image = load_detection_evaluation_adversarial_image(
            evaluation_id=evaluation_id,
            image_id=image_id,
        )
        if original_image.size != adversarial_image.size:
            raise BadRequestError("Las geometrías de las imágenes de la evaluación de detección no coinciden")

        original_eigencam = _draw_stored_predictions(
            await run_detection_model_operation(
                generate_detection_eigencam,
                model,
                original_image,
            ),
            image_result.original_detections,
        )
        adversarial_eigencam = _draw_stored_predictions(
            await run_detection_model_operation(
                generate_detection_eigencam,
                model,
                adversarial_image,
            ),
            image_result.adversarial_detections,
        )
        if (
            original_eigencam.size != original_image.size
            or adversarial_eigencam.size != original_image.size
        ):
            raise ValueError(
                "Generated detection EigenCAM geometry does not match the source"
            )

        await asyncio.to_thread(
            save_detection_eigencam_pair,
            evaluation_id=evaluation_id,
            image_id=image_id,
            original_eigencam=original_eigencam,
            adversarial_eigencam=adversarial_eigencam,
        )

    return _build_response(evaluation_id, image_id)


def _resolve_image_result(
    results: list[DetectionEvaluationImageResult],
    image_id: str,
) -> DetectionEvaluationImageResult:
    image_result = next(
        (result for result in results if result.image_id == image_id),
        None,
    )
    if image_result is None:
        raise NotFoundError("Imagen de la evaluación de detección no encontrada")
    return image_result


def _draw_stored_predictions(
    image: Image.Image,
    detections: list[DetectionPrediction],
) -> Image.Image:
    output = image.copy().convert("RGB")
    draw = ImageDraw.Draw(output)
    width, height = output.size

    for detection in detections:
        box = detection.bbox
        x1 = max(0.0, min(float(width), box.x1))
        y1 = max(0.0, min(float(height), box.y1))
        x2 = max(0.0, min(float(width), box.x2))
        y2 = max(0.0, min(float(height), box.y2))
        if x1 >= x2 or y1 >= y2:
            continue

        draw.rectangle([(x1, y1), (x2, y2)], outline=(0, 255, 255), width=3)
        label = f"{detection.label} {detection.confidence:.2f}"
        text_y = max(0.0, y1 - 18.0)
        text_box = draw.textbbox((x1, text_y), label)
        draw.rectangle(text_box, fill=(0, 255, 255))
        draw.text((x1, text_y), label, fill=(0, 0, 0))

    return output


def _build_response(
    evaluation_id: str,
    image_id: str,
) -> DetectionEvaluationEigenCamResponse:
    filenames = detection_evaluation_image_artifact_filenames(
        evaluation_id,
        image_id,
    )
    return DetectionEvaluationEigenCamResponse(
        evaluation_id=evaluation_id,
        image_id=image_id,
        original_eigencam_image_url=(
            f"/results/{evaluation_id}/{filenames['original_eigencam']}"
        ),
        adversarial_eigencam_image_url=(
            f"/results/{evaluation_id}/{filenames['adversarial_eigencam']}"
        ),
    )

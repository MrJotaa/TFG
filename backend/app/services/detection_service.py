from core.cancellation import CancellationToken
from core.task_compatibility import (
    DETECTION_TASK_TYPE,
    require_model_task_type,
)
from schemas.detection import (
    DetectionRequest,
    DetectionResponse,
    to_detection_predictions,
)
from services.dataset_service import get_image_from_dataset
from services.detection_model_execution import run_detection_model_operation
from services.model_service import get_active_model_instance
from utils.result_files import (
    create_result_directory,
    save_result_image,
    save_result_metadata,
)


async def detect_objects(
    request: DetectionRequest,
    *,
    cancellation_token: CancellationToken | None = None,
    owner_id: str | None = None,
) -> DetectionResponse:
    image = await get_image_from_dataset(
        dataset_id=request.dataset_id,
        image_id=request.image_id,
        owner_id=owner_id,
    )

    model = await run_detection_model_operation(get_active_model_instance)

    require_model_task_type(
        model=model,
        expected_task_type=DETECTION_TASK_TYPE,
        endpoint="/detect",
    )

    if cancellation_token is not None:
        cancellation_token.raise_if_cancelled()
    detection_result, annotated_image = await run_detection_model_operation(
        model.detect_with_visualization,
        image,
    )
    if cancellation_token is not None:
        cancellation_token.raise_if_cancelled()

    result_id, result_path = create_result_directory(prefix="detection")

    annotated_image_url = save_result_image(
        image=annotated_image,
        result_id=result_id,
        result_path=result_path,
        filename="annotated.png",
    )

    save_result_metadata(
        result_path=result_path,
        metadata={
            "owner_id": owner_id,
            "result_id": result_id,
            "result_type": "detection",
            "dataset_id": request.dataset_id,
            "image_id": request.image_id,
            "model_id": model.id,
            "model_name": model.name,
            "detections_count": len(detection_result.detections),
            "expires_in_hours": 24,
        },
    )

    return DetectionResponse(
        model_id=model.id,
        model_name=model.name,
        detections=to_detection_predictions(detection_result.detections),
        annotated_image_url=annotated_image_url,
    )

from core.task_compatibility import (
    CLASSIFICATION_TASK_TYPE,
    require_model_task_type,
)
from schemas.predict import PredictResponse
from services.dataset_service import get_image_from_dataset
from services.model_service import get_active_model_instance


async def predict_image(
    dataset_id: str,
    image_id: str,
) -> PredictResponse:
    """
    Realiza una predicción sobre una imagen persistida en un dataset
    usando el modelo activo.
    """
    image = await get_image_from_dataset(
        dataset_id=dataset_id,
        image_id=image_id,
    )

    model = get_active_model_instance()

    require_model_task_type(
        model=model,
        expected_task_type=CLASSIFICATION_TASK_TYPE,
        endpoint="/predict",
    )

    return model.predict(image)

from fastapi import HTTPException

from schemas.predict import PredictResponse
from services.dataset_service import get_image_from_dataset
from services.model_service import get_active_model_instance
from core.task_compatibility import (
    CLASSIFICATION_TASK_TYPE,
    require_model_task_type,
)


async def predict_image(dataset_id: str, image_id: str) -> PredictResponse:
    """
    Realiza una predicción sobre una imagen persistida en un dataset
    usando el modelo activo.

    La lógica concreta de preprocesado e inferencia queda delegada
    al modelo seleccionado.
    """
    try:
        image = await get_image_from_dataset(dataset_id, image_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    try:
        model = get_active_model_instance()

        require_model_task_type(
            model=model,
            expected_task_type=CLASSIFICATION_TASK_TYPE,
            endpoint="/predict",
        )

        return model.predict(image)
    except ValueError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        )

    return model.predict(image)
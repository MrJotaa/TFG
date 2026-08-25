from core.errors import BadRequestError

CLASSIFICATION_TASK_TYPE = "classification"
DETECTION_TASK_TYPE = "detection"

TASK_TYPE_LABELS = {
    CLASSIFICATION_TASK_TYPE: "clasificación",
    DETECTION_TASK_TYPE: "detección",
}


def _task_type_label(task_type: str) -> str:
    return TASK_TYPE_LABELS.get(task_type, task_type)


def require_model_task_type(
    model,
    expected_task_type: str,
    endpoint: str,
) -> None:
    """
    Comprueba que el modelo activo es compatible con el endpoint.

    Args:
        model: instancia del modelo activo.
        expected_task_type: tipo de tarea requerido por el endpoint.
        endpoint: ruta utilizada para construir un mensaje uniforme.

    Raises:
        BadRequestError: si el modelo no pertenece al tipo esperado.
    """
    actual_task_type = getattr(model, "task_type", None)

    if actual_task_type != expected_task_type:
        task_label = _task_type_label(expected_task_type)
        raise BadRequestError(
            f"El endpoint '{endpoint}' requiere un modelo de {task_label}"
        )


def require_attack_task_type(
    attack,
    task_type: str,
) -> None:
    """
    Comprueba que un ataque admite el tipo de tarea del modelo.

    Raises:
        BadRequestError: si el ataque no es compatible.
    """
    if not attack.supports_task_type(task_type):
        task_label = _task_type_label(task_type)
        raise BadRequestError(
            f"El ataque '{attack.id}' no es compatible con la tarea de {task_label}"
        )

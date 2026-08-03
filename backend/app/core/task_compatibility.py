from core.errors import BadRequestError

CLASSIFICATION_TASK_TYPE = "classification"
DETECTION_TASK_TYPE = "detection"


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
        raise BadRequestError(
            f"Endpoint '{endpoint}' requires a {expected_task_type} model"
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
        raise BadRequestError(
            f"Attack '{attack.id}' does not support task type '{task_type}'"
        )

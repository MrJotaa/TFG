class ApplicationError(Exception):
    """
    Error controlado producido por la lógica de la aplicación.

    Los manejadores de FastAPI transforman estas excepciones en
    respuestas HTTP con una estructura uniforme.
    """

    status_code = 500
    default_detail = "Error de la aplicación"

    def __init__(self, detail: str | dict | None = None):
        self.detail = detail if detail is not None else self.default_detail

        super().__init__(self.detail)


class BadRequestError(ApplicationError):
    """
    La operación solicitada no puede ejecutarse con los datos
    o el estado actual de la aplicación.
    """

    status_code = 400
    default_detail = "Solicitud no válida"


class UnauthorizedError(ApplicationError):
    """
    La petición no contiene credenciales válidas.
    """

    status_code = 401
    default_detail = "No autorizado"


class ForbiddenError(ApplicationError):
    """El usuario autenticado no puede acceder al recurso solicitado."""

    status_code = 403
    default_detail = "Acceso denegado"


class ConflictError(ApplicationError):
    """El estado del recurso cambió durante una operación concurrente."""

    status_code = 409
    default_detail = "Conflicto"


class NotFoundError(ApplicationError):
    """
    El recurso solicitado no existe.
    """

    status_code = 404
    default_detail = "Recurso no encontrado"


class GoneError(ApplicationError):
    """El recurso existió, pero sus artefactos temporales ya no están disponibles."""

    status_code = 410
    default_detail = "El recurso ya no está disponible"


class ClientDisconnectedError(ApplicationError):
    """El cliente canceló una operación que todavía estaba en ejecución."""

    status_code = 499
    default_detail = "Operación cancelada por el cliente"

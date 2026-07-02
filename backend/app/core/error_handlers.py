import logging

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from core.errors import ApplicationError


logger = logging.getLogger(__name__)


async def application_error_handler(
    request: Request,
    error: ApplicationError,
) -> JSONResponse:
    """
    Convierte errores controlados de la aplicación en respuestas HTTP.

    Estos errores representan situaciones esperadas, como recursos
    inexistentes, credenciales inválidas u operaciones incompatibles.
    """
    return JSONResponse(
        status_code=error.status_code,
        content={
            "detail": error.detail,
        },
    )


async def unexpected_error_handler(
    request: Request,
    error: Exception,
) -> JSONResponse:
    """
    Registra errores inesperados y devuelve una respuesta genérica.

    El detalle interno se conserva en los logs, pero nunca se expone
    al cliente.
    """
    logger.error(
        "Unhandled exception while processing %s %s",
        request.method,
        request.url.path,
        exc_info=(
            type(error),
            error,
            error.__traceback__,
        ),
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Internal server error",
        },
    )


def register_error_handlers(app: FastAPI) -> None:
    """
    Registra los manejadores globales de errores.
    """
    app.add_exception_handler(
        ApplicationError,
        application_error_handler,
    )

    app.add_exception_handler(
        Exception,
        unexpected_error_handler,
    )

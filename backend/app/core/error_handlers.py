from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from core.errors import ApplicationError


async def application_error_handler(
    request: Request,
    error: ApplicationError,
) -> JSONResponse:
    """
    Convierte errores controlados de la aplicación en respuestas HTTP.

    El parámetro request se conserva porque forma parte de la interfaz
    requerida por los manejadores de excepciones de FastAPI.
    """
    return JSONResponse(
        status_code=error.status_code,
        content={
            "detail": error.detail,
        },
    )


def register_error_handlers(app: FastAPI) -> None:
    """
    Registra los manejadores globales de errores de la aplicación.
    """
    app.add_exception_handler(
        ApplicationError,
        application_error_handler,
    )
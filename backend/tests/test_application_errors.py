import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import api.routes.models as models_route
from core.error_handlers import register_error_handlers
from core.errors import (
    ApplicationError,
    BadRequestError,
    NotFoundError,
    UnauthorizedError,
)
from main import app


@pytest.mark.parametrize(
    (
        "error_class",
        "expected_status_code",
        "expected_default_detail",
    ),
    [
        (
            ApplicationError,
            500,
            "Error de la aplicación",
        ),
        (
            BadRequestError,
            400,
            "Solicitud no válida",
        ),
        (
            UnauthorizedError,
            401,
            "No autorizado",
        ),
        (
            NotFoundError,
            404,
            "Recurso no encontrado",
        ),
    ],
)
def test_application_errors_have_expected_defaults(
    error_class,
    expected_status_code,
    expected_default_detail,
):
    error = error_class()

    assert error.status_code == expected_status_code
    assert error.detail == expected_default_detail
    assert str(error) == expected_default_detail


def test_application_error_accepts_custom_detail():
    error = NotFoundError("Dataset not found")

    assert error.status_code == 404
    assert error.detail == "Dataset not found"
    assert str(error) == "Dataset not found"


def test_application_error_handler_is_registered():
    assert ApplicationError in app.exception_handlers


@pytest.mark.parametrize(
    (
        "error_class",
        "expected_status_code",
        "detail",
    ),
    [
        (
            BadRequestError,
            400,
            "Invalid operation",
        ),
        (
            UnauthorizedError,
            401,
            "Invalid credentials",
        ),
        (
            NotFoundError,
            404,
            "Model not found",
        ),
        (
            ApplicationError,
            500,
            "Controlled application error",
        ),
    ],
)
def test_application_errors_are_converted_to_http_responses(
    client,
    monkeypatch,
    error_class,
    expected_status_code,
    detail,
):
    def fake_get_active_model():
        raise error_class(detail)

    monkeypatch.setattr(
        models_route,
        "get_active_model",
        fake_get_active_model,
    )

    response = client.get("/models/active")

    assert response.status_code == expected_status_code
    assert response.json() == {
        "detail": detail,
    }


def test_unexpected_error_handler_is_registered():
    assert Exception in app.exception_handlers


def test_unexpected_error_returns_generic_500_response():
    test_app = FastAPI()

    register_error_handlers(test_app)

    @test_app.get("/unexpected-error")
    async def raise_unexpected_error():
        raise RuntimeError("Sensitive internal database failure")

    with TestClient(
        test_app,
        raise_server_exceptions=False,
    ) as error_client:
        response = error_client.get("/unexpected-error")

    assert response.status_code == 500
    assert response.json() == {
        "detail": "Error interno del servidor",
    }

    assert "Sensitive internal database failure" not in response.text


def test_application_errors_are_not_converted_to_generic_500():
    test_app = FastAPI()

    register_error_handlers(test_app)

    @test_app.get("/controlled-error")
    async def raise_controlled_error():
        raise BadRequestError("Invalid operation")

    with TestClient(
        test_app,
        raise_server_exceptions=False,
    ) as error_client:
        response = error_client.get("/controlled-error")

    assert response.status_code == 400
    assert response.json() == {
        "detail": "Invalid operation",
    }

import pytest

import api.routes.models as models_route
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
            "Application error",
        ),
        (
            BadRequestError,
            400,
            "Bad request",
        ),
        (
            UnauthorizedError,
            401,
            "Unauthorized",
        ),
        (
            NotFoundError,
            404,
            "Resource not found",
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
"""Security contract for the whole FastAPI surface.

This test intentionally discovers routes from the application instead of
maintaining a second list of private endpoints. A newly added route without
a JWT dependency will fail the test, even when no feature-specific test exists.
"""

import os
import re
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.routing import APIRoute
from jose import jwt

from api.dependencies.auth import require_current_user
from main import app

# These are the only intentional exceptions to authentication. The root and
# database health check are operational endpoints, not application data.
PUBLIC_ENDPOINTS = {
    ("GET", "/"),
    ("GET", "/health/db"),
    ("POST", "/auth/login"),
    ("POST", "/auth/signup"),
}


def _has_jwt_dependency(dependant) -> bool:
    """Follow router-level and nested FastAPI dependencies recursively."""
    return any(
        dependency.call is require_current_user or _has_jwt_dependency(dependency)
        for dependency in dependant.dependencies
    )


def _api_routes() -> list[APIRoute]:
    # Framework-owned documentation URLs are plain Starlette Routes rather
    # than APIRoute and are outside the application's business API.
    return [route for route in app.routes if isinstance(route, APIRoute)]


def test_every_business_endpoint_declares_jwt_authentication():
    public_found = set()
    for route in _api_routes():
        for method in route.methods:
            key = (method, route.path)
            if key in PUBLIC_ENDPOINTS:
                public_found.add(key)
                assert not _has_jwt_dependency(route.dependant), (
                    f"Public endpoint unexpectedly protected: {method} {route.path}"
                )
            else:
                assert _has_jwt_dependency(route.dependant), (
                    f"Endpoint without JWT protection: {method} {route.path}"
                )

    # The infrastructure routes are declared directly on the FastAPI app.
    # Login/signup are checked by actual anonymous HTTP requests below, since
    # route introspection alone is not a reliable test of their accessibility.
    assert {("GET", "/"), ("GET", "/health/db")} <= public_found


@pytest.mark.parametrize("path", ["/auth/login", "/auth/signup"])
def test_public_auth_endpoints_accept_anonymous_requests(client, path):
    # Both endpoints require JSON fields, so an empty payload must fail body
    # validation (422), not JWT authentication (401) or routing (404).
    response = client.post(path, json={})
    assert response.status_code == 422, (
        f"Anonymous POST {path} returned {response.status_code}"
    )


PRIVATE_GET_ROUTES = sorted(
    {
        route.path
        for route in _api_routes()
        if "GET" in route.methods and ("GET", route.path) not in PUBLIC_ENDPOINTS
    }
)


@pytest.mark.parametrize("route_path", PRIVATE_GET_ROUTES)
def test_all_private_get_routes_reject_anonymous_requests(client, route_path):
    # Supply a syntactically valid placeholder. Authentication must take
    # precedence over resource lookups, file existence and user-id parsing.
    path = re.sub(r"\{[^{}]+\}", "507f1f77bcf86cd799439011", route_path)
    response = client.get(path)
    assert response.status_code == 401, (
        f"Anonymous GET {route_path} returned {response.status_code}"
    )


@pytest.mark.parametrize("authorization", ["Basic abc123", "Bearer invalid-token"])
def test_other_authentication_forms_do_not_bypass_jwt(client, authorization):
    response = client.get("/datasets", headers={"Authorization": authorization})
    assert response.status_code == 401


@pytest.mark.parametrize("token_kind", ["expired", "incorrect_signature"])
@pytest.mark.parametrize("path", ["/auth/me", "/datasets", "/users"])
def test_invalid_signed_tokens_are_rejected(client, registered_user, token_kind, path):
    expires = datetime.now(timezone.utc) + timedelta(hours=1)
    key = os.environ["JWT_SECRET_KEY"]
    if token_kind == "expired":
        expires = datetime.now(timezone.utc) - timedelta(minutes=5)
    else:
        key = "untrusted-signing-key"

    token = jwt.encode(
        {
            "sub": registered_user["user"]["id"],
            "exp": int(expires.timestamp()),
        },
        key,
        algorithm="HS256",
    )
    response = client.get(path, headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 401

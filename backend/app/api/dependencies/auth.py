from fastapi import Depends, Request
from fastapi.security import (
    HTTPAuthorizationCredentials,
    HTTPBearer,
)

from core.auth_cookies import ARTIFACT_SESSION_COOKIE_NAME
from core.errors import UnauthorizedError
from schemas.users import UserResponse
from services.auth_service import get_artifact_current_user, get_current_user

bearer_scheme = HTTPBearer()


async def require_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> UserResponse:
    return await get_current_user(credentials.credentials)


async def require_artifact_current_user(request: Request) -> UserResponse:
    token = request.cookies.get(ARTIFACT_SESSION_COOKIE_NAME)
    if not token:
        raise UnauthorizedError("Autenticación requerida para acceder al artefacto")
    return await get_artifact_current_user(token)

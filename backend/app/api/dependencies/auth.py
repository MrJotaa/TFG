from fastapi import Depends
from fastapi.security import (
    HTTPAuthorizationCredentials,
    HTTPBearer,
)

from schemas.users import UserResponse
from services.auth_service import get_current_user


bearer_scheme = HTTPBearer()


async def require_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(
        bearer_scheme
    ),
) -> UserResponse:
    return await get_current_user(credentials.credentials)
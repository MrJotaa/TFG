from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from schemas.users import UserResponse
from services.auth_service import get_current_user


bearer_scheme = HTTPBearer()


async def require_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> UserResponse:
    try:
        return await get_current_user(credentials.credentials)
    except ValueError as error:
        raise HTTPException(status_code=401, detail=str(error))
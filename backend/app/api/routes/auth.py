from fastapi import APIRouter, Depends, Response

from api.dependencies.auth import require_current_user
from core.auth_cookies import (
    ARTIFACT_SESSION_COOKIE_NAME,
    ARTIFACT_SESSION_COOKIE_PATH,
    artifact_cookie_secure,
)
from schemas.auth import (
    AuthResponse,
    CurrentUserResponse,
    LoginRequest,
    SignupRequest,
)
from schemas.users import UserResponse
from services.auth_service import create_artifact_token, login, signup
from utils.security import ACCESS_TOKEN_EXPIRE_MINUTES

router = APIRouter()


@router.post("/auth/signup", response_model=AuthResponse)
async def signup_user(
    signup_data: SignupRequest,
    response: Response,
) -> AuthResponse:
    auth_response = await signup(signup_data)
    _set_artifact_session_cookie(response, create_artifact_token(auth_response.user))
    return auth_response


@router.post("/auth/login", response_model=AuthResponse)
async def login_user(
    login_data: LoginRequest,
    response: Response,
) -> AuthResponse:
    auth_response = await login(login_data)
    _set_artifact_session_cookie(response, create_artifact_token(auth_response.user))
    return auth_response


@router.get("/auth/me", response_model=CurrentUserResponse)
async def get_me(
    response: Response,
    current_user: UserResponse = Depends(require_current_user),
) -> CurrentUserResponse:
    artifact_token = create_artifact_token(current_user)
    _set_artifact_session_cookie(response, artifact_token)
    return CurrentUserResponse(user=current_user)


@router.post("/auth/logout", status_code=204)
async def logout_user(response: Response) -> Response:
    response.status_code = 204
    response.delete_cookie(
        key=ARTIFACT_SESSION_COOKIE_NAME,
        path=ARTIFACT_SESSION_COOKIE_PATH,
        secure=artifact_cookie_secure(),
        httponly=True,
        samesite="lax",
    )
    return response


def _set_artifact_session_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        key=ARTIFACT_SESSION_COOKIE_NAME,
        value=token,
        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        path=ARTIFACT_SESSION_COOKIE_PATH,
        secure=artifact_cookie_secure(),
        httponly=True,
        samesite="lax",
    )

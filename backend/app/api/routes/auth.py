from typing import Annotated

from fastapi import APIRouter, Depends

from api.dependencies.auth import require_current_user
from schemas.auth import (
    AuthResponse,
    CurrentUserResponse,
    LoginRequest,
    SignupRequest,
)
from schemas.users import UserResponse
from services.auth_service import login, signup

router = APIRouter()


@router.post("/auth/signup")
async def signup_user(
    signup_data: SignupRequest,
) -> AuthResponse:
    return await signup(signup_data)


@router.post("/auth/login")
async def login_user(
    login_data: LoginRequest,
) -> AuthResponse:
    return await login(login_data)


@router.get("/auth/me")
async def get_me(
    current_user: Annotated[UserResponse, Depends(require_current_user)],
) -> CurrentUserResponse:
    return CurrentUserResponse(user=current_user)

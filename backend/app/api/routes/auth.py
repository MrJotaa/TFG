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


@router.post("/auth/signup", response_model=AuthResponse)
async def signup_user(
    signup_data: SignupRequest,
) -> AuthResponse:
    return await signup(signup_data)


@router.post("/auth/login", response_model=AuthResponse)
async def login_user(
    login_data: LoginRequest,
) -> AuthResponse:
    return await login(login_data)


@router.get("/auth/me", response_model=CurrentUserResponse)
async def get_me(
    current_user: UserResponse = Depends(require_current_user),
) -> CurrentUserResponse:
    return CurrentUserResponse(user=current_user)
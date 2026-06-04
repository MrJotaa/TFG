from fastapi import APIRouter, Depends, HTTPException

from api.dependencies.auth import require_current_user
from schemas.auth import AuthResponse, CurrentUserResponse, LoginRequest, SignupRequest
from schemas.users import UserResponse
from services.auth_service import login, signup


router = APIRouter()


@router.post("/auth/signup", response_model=AuthResponse)
async def signup_user(signup_data: SignupRequest) -> AuthResponse:
    try:
        return await signup(signup_data)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))


@router.post("/auth/login", response_model=AuthResponse)
async def login_user(login_data: LoginRequest) -> AuthResponse:
    try:
        return await login(login_data)
    except ValueError as error:
        raise HTTPException(status_code=401, detail=str(error))


@router.get("/auth/me", response_model=CurrentUserResponse)
async def get_me(current_user: UserResponse = Depends(require_current_user)) -> CurrentUserResponse:
    return CurrentUserResponse(user=current_user)
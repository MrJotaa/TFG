from fastapi import APIRouter, HTTPException

from schemas.users import UserCreate, UserResponse, UserUpdate
from services.user_service import (
    create_user,
    delete_user,
    get_user,
    list_users,
    update_user,
)

router = APIRouter()


@router.get("/users", response_model=list[UserResponse])
async def get_users() -> list[UserResponse]:
    return list_users()


@router.post("/users", response_model=UserResponse)
async def register_user(user: UserCreate) -> UserResponse:
    return create_user(user)


@router.get("/users/{user_id}", response_model=UserResponse)
async def get_user_by_id(user_id: int) -> UserResponse:
    try:
        return get_user(user_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.patch("/users/{user_id}", response_model=UserResponse)
async def update_user_by_id(user_id: int, user: UserUpdate) -> UserResponse:
    try:
        return update_user(user_id, user)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.delete("/users/{user_id}")
async def delete_user_by_id(user_id: int) -> dict:
    try:
        delete_user(user_id)
        return {"message": "User deleted successfully"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
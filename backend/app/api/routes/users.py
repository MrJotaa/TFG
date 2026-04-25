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
    return await list_users()


@router.post("/users", response_model=UserResponse)
async def register_user(user: UserCreate) -> UserResponse:
    try:
        return await create_user(user)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/users/{user_id}", response_model=UserResponse)
async def get_user_by_id(user_id: str) -> UserResponse:
    try:
        return await get_user(user_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.patch("/users/{user_id}", response_model=UserResponse)
async def update_user_by_id(user_id: str, user: UserUpdate) -> UserResponse:
    try:
        return await update_user(user_id, user)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.delete("/users/{user_id}")
async def delete_user_by_id(user_id: str) -> dict:
    try:
        await delete_user(user_id)
        return {"message": "User deleted successfully"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
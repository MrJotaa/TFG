from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from schemas.datasets import DatasetResponse
from services.dataset_service import (
    create_dataset_from_images,
    get_dataset,
    list_datasets,
)

router = APIRouter()


@router.post("/datasets", response_model=DatasetResponse)
async def upload_dataset(
    user_id: str = Form(...),
    name: str = Form(...),
    files: list[UploadFile] = File(...),
) -> DatasetResponse:
    try:
        return await create_dataset_from_images(user_id, name, files)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/datasets", response_model=list[DatasetResponse])
async def get_datasets() -> list[DatasetResponse]:
    return await list_datasets()


@router.get("/datasets/{dataset_id}", response_model=DatasetResponse)
async def get_dataset_by_id(dataset_id: str) -> DatasetResponse:
    try:
        return await get_dataset(dataset_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from schemas.datasets import DatasetResponse, DatasetImage
from services.dataset_service import (
    create_dataset_from_images,
    get_dataset,
    list_datasets,
    delete_dataset,
    add_images_to_dataset,
    delete_image_from_dataset,
    get_image_file_response,
    list_images_in_dataset,

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

@router.delete("/datasets/{dataset_id}")
async def delete_dataset_by_id(dataset_id: str) -> dict:
    try:
        await delete_dataset(dataset_id)
        return {"message": "Dataset deleted successfully"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/datasets/{dataset_id}/images", response_model=list[DatasetImage])
async def get_dataset_images(dataset_id: str) -> list[DatasetImage]:
    try:
        return await list_images_in_dataset(dataset_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/datasets/{dataset_id}/images", response_model=list[DatasetImage])
async def add_images_to_existing_dataset(
    dataset_id: str,
    files: list[UploadFile] = File(...),
) -> list[DatasetImage]:
    try:
        return await add_images_to_dataset(dataset_id, files)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/datasets/{dataset_id}/images/{image_id}")
async def get_dataset_image(dataset_id: str, image_id: str) -> FileResponse:
    try:
        return await get_image_file_response(dataset_id, image_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.delete("/datasets/{dataset_id}/images/{image_id}")
async def delete_dataset_image(dataset_id: str, image_id: str) -> dict:
    try:
        await delete_image_from_dataset(dataset_id, image_id)
        return {"message": "Image deleted successfully"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
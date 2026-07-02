from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import FileResponse

from schemas.datasets import (
    DatasetImage,
    DatasetImageLabelUpdate,
    DatasetResponse,
)
from services.dataset_service import (
    add_images_to_dataset,
    create_dataset_from_images,
    delete_dataset,
    delete_image_from_dataset,
    get_dataset,
    get_image_file_response,
    list_datasets,
    list_images_in_dataset,
    update_image_label,
)


router = APIRouter()


@router.post("/datasets", response_model=DatasetResponse)
async def upload_dataset(
    user_id: str = Form(...),
    name: str = Form(...),
    files: list[UploadFile] = File(...),
) -> DatasetResponse:
    return await create_dataset_from_images(user_id, name, files)


@router.get("/datasets", response_model=list[DatasetResponse])
async def get_datasets() -> list[DatasetResponse]:
    return await list_datasets()


@router.get("/datasets/{dataset_id}", response_model=DatasetResponse)
async def get_dataset_by_id(dataset_id: str) -> DatasetResponse:
    return await get_dataset(dataset_id)


@router.delete("/datasets/{dataset_id}")
async def delete_dataset_by_id(dataset_id: str) -> dict:
    await delete_dataset(dataset_id)
    return {"message": "Dataset deleted successfully"}


@router.get(
    "/datasets/{dataset_id}/images",
    response_model=list[DatasetImage],
)
async def get_dataset_images(
    dataset_id: str,
) -> list[DatasetImage]:
    return await list_images_in_dataset(dataset_id)


@router.post(
    "/datasets/{dataset_id}/images",
    response_model=list[DatasetImage],
)
async def add_images_to_existing_dataset(
    dataset_id: str,
    files: list[UploadFile] = File(...),
) -> list[DatasetImage]:
    return await add_images_to_dataset(dataset_id, files)


@router.get("/datasets/{dataset_id}/images/{image_id}")
async def get_dataset_image(
    dataset_id: str,
    image_id: str,
) -> FileResponse:
    return await get_image_file_response(dataset_id, image_id)


@router.delete("/datasets/{dataset_id}/images/{image_id}")
async def delete_dataset_image(
    dataset_id: str,
    image_id: str,
) -> dict:
    await delete_image_from_dataset(dataset_id, image_id)
    return {"message": "Image deleted successfully"}


@router.patch(
    "/datasets/{dataset_id}/images/{image_id}/label",
    response_model=DatasetImage,
)
async def update_dataset_image_label(
    dataset_id: str,
    image_id: str,
    label_update: DatasetImageLabelUpdate,
) -> DatasetImage:
    return await update_image_label(
        dataset_id,
        image_id,
        label_update,
    )
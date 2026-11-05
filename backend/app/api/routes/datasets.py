from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.responses import FileResponse

from api.dependencies.auth import require_current_user
from core.errors import ForbiddenError
from schemas.datasets import (
    CocoAnnotationImportResponse,
    DatasetAnnotationCoverage,
    DatasetImage,
    DatasetImageLabelUpdate,
    DatasetLabelResolutionResponse,
    DatasetMetadataUpdate,
    DatasetResponse,
    DatasetTaskType,
)
from schemas.users import UserResponse
from services.classification_label_service import (
    get_active_dataset_label_resolution,
)
from services.dataset_annotation_service import (
    get_annotation_coverage,
    import_coco_annotations,
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
    update_dataset_metadata,
    update_image_label,
)

router = APIRouter()


@router.post("/datasets")
async def upload_dataset(
    user_id: str | None = Form(None),
    name: str = Form(...),
    task_type: DatasetTaskType | None = Form(None),
    files: list[UploadFile] = File(...),
    current_user: UserResponse = Depends(require_current_user),
) -> DatasetResponse:
    if user_id is not None and user_id != current_user.id:
        raise ForbiddenError("No puedes crear datasets para otro usuario")
    return await create_dataset_from_images(current_user.id, name, files, task_type)


@router.get("/datasets")
async def get_datasets(
    current_user: UserResponse = Depends(require_current_user),
) -> list[DatasetResponse]:
    return await list_datasets(owner_id=current_user.id)


@router.get("/datasets/{dataset_id}")
async def get_dataset_by_id(
    dataset_id: str,
    current_user: UserResponse = Depends(require_current_user),
) -> DatasetResponse:
    return await get_dataset(dataset_id, owner_id=current_user.id)


@router.patch("/datasets/{dataset_id}")
async def update_dataset_by_id(
    dataset_id: str,
    metadata: DatasetMetadataUpdate,
    current_user: UserResponse = Depends(require_current_user),
) -> DatasetResponse:
    return await update_dataset_metadata(dataset_id, metadata, owner_id=current_user.id)


@router.get(
    "/datasets/{dataset_id}/label-resolution",
)
async def get_dataset_label_resolution(
    dataset_id: str,
    current_user: UserResponse = Depends(require_current_user),
) -> DatasetLabelResolutionResponse:
    return await get_active_dataset_label_resolution(
        dataset_id, owner_id=current_user.id
    )


@router.get(
    "/datasets/{dataset_id}/annotations/coverage",
)
async def get_dataset_annotation_coverage(
    dataset_id: str,
    current_user: UserResponse = Depends(require_current_user),
) -> DatasetAnnotationCoverage:
    return await get_annotation_coverage(dataset_id, current_user)


@router.post(
    "/datasets/{dataset_id}/annotations/coco",
)
async def import_dataset_coco_annotations(
    dataset_id: str,
    file: UploadFile = File(...),
    current_user: UserResponse = Depends(require_current_user),
) -> CocoAnnotationImportResponse:
    return await import_coco_annotations(dataset_id, file, current_user)


@router.delete("/datasets/{dataset_id}")
async def delete_dataset_by_id(
    dataset_id: str,
    current_user: UserResponse = Depends(require_current_user),
) -> dict:
    await delete_dataset(dataset_id, owner_id=current_user.id)
    return {"message": "Conjunto de datos eliminado correctamente"}


@router.get(
    "/datasets/{dataset_id}/images",
)
async def get_dataset_images(
    dataset_id: str,
    current_user: UserResponse = Depends(require_current_user),
) -> list[DatasetImage]:
    return await list_images_in_dataset(dataset_id, owner_id=current_user.id)


@router.post(
    "/datasets/{dataset_id}/images",
)
async def add_images_to_existing_dataset(
    dataset_id: str,
    files: list[UploadFile] = File(...),
    current_user: UserResponse = Depends(require_current_user),
) -> list[DatasetImage]:
    return await add_images_to_dataset(dataset_id, files, owner_id=current_user.id)


@router.get("/datasets/{dataset_id}/images/{image_id}")
async def get_dataset_image(
    dataset_id: str,
    image_id: str,
    current_user: UserResponse = Depends(require_current_user),
) -> FileResponse:
    return await get_image_file_response(
        dataset_id, image_id, owner_id=current_user.id
    )


@router.delete("/datasets/{dataset_id}/images/{image_id}")
async def delete_dataset_image(
    dataset_id: str,
    image_id: str,
    current_user: UserResponse = Depends(require_current_user),
) -> dict:
    await delete_image_from_dataset(dataset_id, image_id, owner_id=current_user.id)
    return {"message": "Imagen eliminada correctamente"}


@router.patch(
    "/datasets/{dataset_id}/images/{image_id}/label",
)
async def update_dataset_image_label(
    dataset_id: str,
    image_id: str,
    label_update: DatasetImageLabelUpdate,
    current_user: UserResponse = Depends(require_current_user),
) -> DatasetImage:
    return await update_image_label(
        dataset_id,
        image_id,
        label_update,
        owner_id=current_user.id,
    )

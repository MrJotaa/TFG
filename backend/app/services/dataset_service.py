import mimetypes
import os
import shutil
import uuid

from bson import ObjectId
from fastapi import UploadFile
from fastapi.responses import FileResponse
from PIL import Image

from core.errors import BadRequestError, NotFoundError
from db.collections import (
    annotation_categories_collection,
    datasets_collection,
    object_annotations_collection,
    users_collection,
)
from schemas.datasets import (
    DatasetImage,
    DatasetImageLabelUpdate,
    DatasetResponse,
)
from utils.file_manager import (
    create_directory,
    is_valid_image_file,
    save_upload_file,
)


UPLOADS_DATASETS_PATH = os.getenv(
    "UPLOADS_DATASETS_PATH",
    "uploads/datasets",
)

INVALID_DATASET_ID_MESSAGE = "Invalid dataset id"
INVALID_USER_ID_MESSAGE = "Invalid user id"
DATASET_NOT_FOUND_MESSAGE = "Dataset not found"
USER_NOT_FOUND_MESSAGE = "User not found"
IMAGE_NOT_FOUND_MESSAGE = "Image not found"
IMAGE_FILE_NOT_FOUND_MESSAGE = "Image file not found"


def _to_dataset_image(
    dataset_id: str,
    image_metadata: dict,
) -> DatasetImage:
    return DatasetImage(
        id=image_metadata["id"],
        filename=image_metadata["filename"],
        url=(f"/datasets/{dataset_id}/images/{image_metadata['id']}"),
        label=image_metadata.get("label"),
        width=image_metadata.get("width"),
        height=image_metadata.get("height"),
        annotation_review_status=image_metadata.get(
            "annotation_review_status",
            "unreviewed",
        ),
    )


def dataset_to_response(dataset: dict) -> DatasetResponse:
    dataset_id = str(dataset["_id"])

    return DatasetResponse(
        id=dataset_id,
        name=dataset["name"],
        owner_id=str(dataset["owner_id"]),
        image_count=dataset["image_count"],
        images=[_to_dataset_image(dataset_id, image) for image in dataset["images"]],
    )


def _validate_dataset_id(dataset_id: str) -> ObjectId:
    if not ObjectId.is_valid(dataset_id):
        raise BadRequestError(INVALID_DATASET_ID_MESSAGE)

    return ObjectId(dataset_id)


def _validate_user_id(user_id: str) -> ObjectId:
    if not ObjectId.is_valid(user_id):
        raise BadRequestError(INVALID_USER_ID_MESSAGE)

    return ObjectId(user_id)


async def _get_dataset_document(
    dataset_id: str,
) -> tuple[ObjectId, dict]:
    object_id = _validate_dataset_id(dataset_id)
    dataset = await datasets_collection.find_one({"_id": object_id})

    if not dataset:
        raise NotFoundError(DATASET_NOT_FOUND_MESSAGE)

    return object_id, dataset


def _get_image_metadata(dataset: dict, image_id: str) -> dict:
    image_metadata = next(
        (image for image in dataset["images"] if image["id"] == image_id),
        None,
    )

    if not image_metadata:
        raise NotFoundError(IMAGE_NOT_FOUND_MESSAGE)

    return image_metadata


def _read_authoritative_image_dimensions(image_path: str) -> tuple[int, int]:
    try:
        with Image.open(image_path) as image:
            width, height = image.size
    except (OSError, ValueError) as error:
        raise BadRequestError("Uploaded file is not a valid image") from error

    if width <= 0 or height <= 0:
        raise BadRequestError("Uploaded image dimensions are invalid")

    return width, height


async def create_dataset_from_images(
    user_id: str,
    name: str,
    files: list[UploadFile],
) -> DatasetResponse:
    user_object_id = _validate_user_id(user_id)
    user = await users_collection.find_one({"_id": user_object_id})

    if not user:
        raise NotFoundError(USER_NOT_FOUND_MESSAGE)

    if not files:
        raise BadRequestError("Dataset must contain at least one image")

    for file in files:
        if not is_valid_image_file(file.filename):
            raise BadRequestError(f"Invalid image file: {file.filename}")

    dataset_id = ObjectId()
    dataset_path = os.path.join(
        UPLOADS_DATASETS_PATH,
        f"user_{user_id}",
        f"dataset_{dataset_id}",
    )

    create_directory(dataset_path)

    images_metadata = []

    try:
        for file in files:
            image_id = str(uuid.uuid4())
            extension = os.path.splitext(file.filename)[1].lower()
            stored_filename = f"{image_id}{extension}"
            image_path = os.path.join(dataset_path, stored_filename)

            save_upload_file(file, image_path)
            width, height = _read_authoritative_image_dimensions(image_path)

            images_metadata.append(
                {
                    "id": image_id,
                    "filename": file.filename,
                    "original_file_name": file.filename,
                    "stored_filename": stored_filename,
                    "path": image_path,
                    "label": None,
                    "width": width,
                    "height": height,
                    "annotation_review_status": "unreviewed",
                }
            )
    except (BadRequestError, OSError):
        shutil.rmtree(dataset_path, ignore_errors=True)
        raise

    dataset_doc = {
        "_id": dataset_id,
        "name": name,
        "owner_id": user_object_id,
        "image_count": len(images_metadata),
        "path": dataset_path,
        "images": images_metadata,
        "dataset_revision": 0,
    }

    await datasets_collection.insert_one(dataset_doc)

    await users_collection.update_one(
        {"_id": user_object_id},
        {"$addToSet": {"datasets": dataset_id}},
    )

    created_dataset = await datasets_collection.find_one({"_id": dataset_id})

    return dataset_to_response(created_dataset)


async def list_datasets() -> list[DatasetResponse]:
    datasets = []

    async for dataset in datasets_collection.find():
        datasets.append(dataset_to_response(dataset))

    return datasets


async def get_dataset(dataset_id: str) -> DatasetResponse:
    _, dataset = await _get_dataset_document(dataset_id)
    return dataset_to_response(dataset)


async def list_images_in_dataset(
    dataset_id: str,
) -> list[DatasetImage]:
    _, dataset = await _get_dataset_document(dataset_id)

    return [_to_dataset_image(dataset_id, image) for image in dataset["images"]]


async def get_image_file_response(
    dataset_id: str,
    image_id: str,
) -> FileResponse:
    _, dataset = await _get_dataset_document(dataset_id)
    image_metadata = _get_image_metadata(dataset, image_id)
    image_path = image_metadata["path"]

    if not os.path.exists(image_path):
        raise NotFoundError(IMAGE_FILE_NOT_FOUND_MESSAGE)

    media_type, _ = mimetypes.guess_type(image_path)

    return FileResponse(
        path=image_path,
        filename=image_metadata["filename"],
        media_type=media_type or "application/octet-stream",
    )


async def add_images_to_dataset(
    dataset_id: str,
    files: list[UploadFile],
) -> list[DatasetImage]:
    object_id, dataset = await _get_dataset_document(dataset_id)

    if not files:
        raise BadRequestError("At least one image must be provided")

    images_metadata = []

    created_paths = []
    try:
        for file in files:
            if not is_valid_image_file(file.filename):
                raise BadRequestError(f"Invalid image file: {file.filename}")

            image_id = str(uuid.uuid4())
            extension = os.path.splitext(file.filename)[1].lower()
            stored_filename = f"{image_id}{extension}"
            image_path = os.path.join(dataset["path"], stored_filename)

            save_upload_file(file, image_path)
            created_paths.append(image_path)
            width, height = _read_authoritative_image_dimensions(image_path)

            images_metadata.append(
                {
                    "id": image_id,
                    "filename": file.filename,
                    "original_file_name": file.filename,
                    "stored_filename": stored_filename,
                    "path": image_path,
                    "label": None,
                    "width": width,
                    "height": height,
                    "annotation_review_status": "unreviewed",
                }
            )
    except (BadRequestError, OSError):
        for created_path in created_paths:
            if os.path.exists(created_path):
                os.remove(created_path)
        raise

    await datasets_collection.update_one(
        {"_id": object_id},
        {
            "$push": {"images": {"$each": images_metadata}},
            "$inc": {
                "image_count": len(images_metadata),
                "dataset_revision": 1,
            },
        },
    )

    return [_to_dataset_image(dataset_id, image) for image in images_metadata]


async def delete_image_from_dataset(
    dataset_id: str,
    image_id: str,
) -> None:
    object_id, dataset = await _get_dataset_document(dataset_id)
    image_metadata = _get_image_metadata(dataset, image_id)
    image_path = image_metadata["path"]

    if os.path.exists(image_path):
        os.remove(image_path)

    await datasets_collection.update_one(
        {"_id": object_id},
        {
            "$pull": {"images": {"id": image_id}},
            "$inc": {
                "image_count": -1,
                "dataset_revision": 1,
            },
        },
    )
    await object_annotations_collection.delete_many(
        {
            "dataset_id": object_id,
            "image_id": image_id,
        }
    )


async def delete_dataset(dataset_id: str) -> None:
    object_id, dataset = await _get_dataset_document(dataset_id)
    dataset_path = dataset["path"]

    if os.path.exists(dataset_path):
        shutil.rmtree(dataset_path)

    await datasets_collection.delete_one({"_id": object_id})
    await annotation_categories_collection.delete_many({"dataset_id": object_id})
    await object_annotations_collection.delete_many({"dataset_id": object_id})

    await users_collection.update_many(
        {"datasets": object_id},
        {"$pull": {"datasets": object_id}},
    )


async def get_image_from_dataset(
    dataset_id: str,
    image_id: str,
) -> Image.Image:
    _, dataset = await _get_dataset_document(dataset_id)
    image_metadata = _get_image_metadata(dataset, image_id)
    image_path = image_metadata["path"]

    if not os.path.exists(image_path):
        raise NotFoundError(IMAGE_FILE_NOT_FOUND_MESSAGE)

    return Image.open(image_path).convert("RGB")


async def get_datasets_by_user(
    user_id: str,
) -> list[DatasetResponse]:
    user_object_id = _validate_user_id(user_id)
    user = await users_collection.find_one({"_id": user_object_id})

    if not user:
        raise NotFoundError(USER_NOT_FOUND_MESSAGE)

    datasets = []

    async for dataset in datasets_collection.find({"owner_id": user_object_id}):
        datasets.append(dataset_to_response(dataset))

    return datasets


async def update_image_label(
    dataset_id: str,
    image_id: str,
    label_update: DatasetImageLabelUpdate,
) -> DatasetImage:
    object_id, dataset = await _get_dataset_document(dataset_id)
    image_metadata = _get_image_metadata(dataset, image_id)

    normalized_label = (
        label_update.label.strip()
        if label_update.label and label_update.label.strip()
        else None
    )

    await datasets_collection.update_one(
        {
            "_id": object_id,
            "images.id": image_id,
        },
        {
            "$set": {
                "images.$.label": normalized_label,
            },
            "$inc": {"dataset_revision": 1},
        },
    )

    updated_image_metadata = {
        **image_metadata,
        "label": normalized_label,
    }

    return _to_dataset_image(
        dataset_id,
        updated_image_metadata,
    )

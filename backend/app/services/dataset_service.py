import mimetypes
import shutil
import os
import uuid

from bson import ObjectId
from fastapi import UploadFile
from fastapi.responses import FileResponse
from PIL import Image

from db.collections import datasets_collection, users_collection
from schemas.datasets import DatasetImage, DatasetResponse
from utils.file_manager import (
    create_directory,
    is_valid_image_file,
    save_upload_file,
)

UPLOADS_DATASETS_PATH = "uploads/datasets"

def dataset_to_response(dataset: dict) -> DatasetResponse:
    dataset_id = str(dataset["_id"])

    return DatasetResponse(
        id=dataset_id,
        name=dataset["name"],
        owner_id=str(dataset["owner_id"]),
        image_count=dataset["image_count"],
        images=[
            DatasetImage(
                id=image["id"],
                filename=image["filename"],
                url=f"/datasets/{dataset_id}/images/{image['id']}",
            )
            for image in dataset["images"]
        ],
    )

async def create_dataset_from_images(
    user_id: str,
    name: str,
    files: list[UploadFile],
) -> DatasetResponse:
    if not ObjectId.is_valid(user_id):
        raise ValueError("Invalid user id")

    user = await users_collection.find_one({"_id": ObjectId(user_id)})

    if not user:
        raise ValueError(f"User with id {user_id} not found")

    if not files:
        raise ValueError("Dataset must contain at least one image")

    valid_files = []

    for file in files:
        if not is_valid_image_file(file.filename):
            raise ValueError(f"Invalid image file: {file.filename}")
        valid_files.append(file)

    dataset_id = ObjectId()
    dataset_path = os.path.join(
        UPLOADS_DATASETS_PATH,
        f"user_{user_id}",
        f"dataset_{str(dataset_id)}",
    )

    create_directory(dataset_path)

    images_metadata = []

    for file in valid_files:
        """
        Para evitar colisiones de nombres,
        generamos un ID único para cada imagen
        y lo almacenamos con la extensión original.
        """
        image_id = str(uuid.uuid4())
        extension = os.path.splitext(file.filename)[1].lower()
        stored_filename = f"{image_id}{extension}"
        image_path = os.path.join(dataset_path, stored_filename)

        save_upload_file(file, image_path)

        images_metadata.append(
            {
                "id": image_id,
                "filename": file.filename,
                "stored_filename": stored_filename,
                "path": image_path,
            }
        )

    dataset_doc = {
        "_id": dataset_id,
        "name": name,
        "owner_id": ObjectId(user_id),
        "image_count": len(images_metadata),
        "path": dataset_path,
        "images": images_metadata,
    }

    await datasets_collection.insert_one(dataset_doc)

    await users_collection.update_one(
        {"_id": ObjectId(user_id)},
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
    if not ObjectId.is_valid(dataset_id):
        raise ValueError("Invalid dataset id")

    dataset = await datasets_collection.find_one({"_id": ObjectId(dataset_id)})

    if not dataset:
        raise ValueError(f"Dataset with id {dataset_id} not found")

    return dataset_to_response(dataset)

async def list_images_in_dataset(dataset_id: str) -> list[DatasetImage]:
    if not ObjectId.is_valid(dataset_id):
        raise ValueError("Invalid dataset id")

    dataset = await datasets_collection.find_one({"_id": ObjectId(dataset_id)})

    if not dataset:
        raise ValueError(f"Dataset with id {dataset_id} not found")

    return [
        DatasetImage(
            id=image["id"],
            filename=image["filename"],
            url=f"/datasets/{dataset_id}/images/{image['id']}",
        )
        for image in dataset["images"]
    ]

async def get_image_file_response(dataset_id: str, image_id: str) -> FileResponse:
    if not ObjectId.is_valid(dataset_id):
        raise ValueError("Invalid dataset id")

    dataset = await datasets_collection.find_one({"_id": ObjectId(dataset_id)})

    if not dataset:
        raise ValueError(f"Dataset with id {dataset_id} not found")

    image_metadata = next(
        (image for image in dataset["images"] if image["id"] == image_id),
        None,
    )

    if not image_metadata:
        raise ValueError(f"Image with id {image_id} not found in dataset {dataset_id}")

    image_path = image_metadata["path"]

    if not os.path.exists(image_path):
        raise ValueError(f"Image file not found on disk: {image_id}")

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
    if not ObjectId.is_valid(dataset_id):
        raise ValueError("Invalid dataset id")

    dataset = await datasets_collection.find_one({"_id": ObjectId(dataset_id)})

    if not dataset:
        raise ValueError(f"Dataset with id {dataset_id} not found")

    if not files:
        raise ValueError("At least one image must be provided")

    images_metadata = []

    for file in files:
        if not is_valid_image_file(file.filename):
            raise ValueError(f"Invalid image file: {file.filename}")

        image_id = str(uuid.uuid4())
        extension = os.path.splitext(file.filename)[1].lower()
        stored_filename = f"{image_id}{extension}"
        image_path = os.path.join(dataset["path"], stored_filename)

        save_upload_file(file, image_path)

        image_metadata = {
            "id": image_id,
            "filename": file.filename,
            "stored_filename": stored_filename,
            "path": image_path,
        }

        images_metadata.append(image_metadata)

    await datasets_collection.update_one(
        {"_id": ObjectId(dataset_id)},
        {
            "$push": {"images": {"$each": images_metadata}},
            "$inc": {"image_count": len(images_metadata)},
        },
    )

    return [
        DatasetImage(
            id=image["id"],
            filename=image["filename"],
            url=f"/datasets/{dataset_id}/images/{image['id']}",
        )
        for image in images_metadata
    ]

async def delete_image_from_dataset(dataset_id: str, image_id: str) -> None:
    if not ObjectId.is_valid(dataset_id):
        raise ValueError("Invalid dataset id")

    dataset = await datasets_collection.find_one({"_id": ObjectId(dataset_id)})

    if not dataset:
        raise ValueError(f"Dataset with id {dataset_id} not found")

    image_metadata = next(
        (image for image in dataset["images"] if image["id"] == image_id),
        None,
    )

    if not image_metadata:
        raise ValueError(f"Image with id {image_id} not found in dataset {dataset_id}")

    image_path = image_metadata["path"]

    if os.path.exists(image_path):
        os.remove(image_path)

    await datasets_collection.update_one(
        {"_id": ObjectId(dataset_id)},
        {
            "$pull": {"images": {"id": image_id}},
            "$inc": {"image_count": -1},
        },
    )

async def delete_dataset(dataset_id: str) -> None:
    if not ObjectId.is_valid(dataset_id):
        raise ValueError("Invalid dataset id")

    dataset = await datasets_collection.find_one({"_id": ObjectId(dataset_id)})

    if not dataset:
        raise ValueError(f"Dataset with id {dataset_id} not found")

    dataset_path = dataset["path"]

    if os.path.exists(dataset_path):
        shutil.rmtree(dataset_path)

    await datasets_collection.delete_one({"_id": ObjectId(dataset_id)})

    await users_collection.update_many(
        {"datasets": ObjectId(dataset_id)},
        {"$pull": {"datasets": ObjectId(dataset_id)}},
    )

# Función adicional para obtener la imagen como objeto PIL, útil para el endpoint de predicción.
async def get_image_from_dataset(dataset_id: str, image_id: str) -> Image.Image:
    if not ObjectId.is_valid(dataset_id):
        raise ValueError("Invalid dataset id")

    dataset = await datasets_collection.find_one({"_id": ObjectId(dataset_id)})

    if not dataset:
        raise ValueError(f"Dataset with id {dataset_id} not found")

    image_metadata = next(
        (image for image in dataset["images"] if image["id"] == image_id),
        None,
    )

    if not image_metadata:
        raise ValueError(f"Image with id {image_id} not found in dataset {dataset_id}")

    image_path = image_metadata["path"]

    if not os.path.exists(image_path):
        raise ValueError(f"Image file not found on disk: {image_id}")

    return Image.open(image_path).convert("RGB")

async def get_datasets_by_user(user_id: str) -> list[DatasetResponse]:
    if not ObjectId.is_valid(user_id):
        raise ValueError("Invalid user id")

    datasets = []

    async for dataset in datasets_collection.find({"owner_id": ObjectId(user_id)}):
        datasets.append(dataset_to_response(dataset))

    return datasets
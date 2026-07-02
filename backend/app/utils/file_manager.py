import os
import shutil
from fastapi import UploadFile

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def is_valid_image_file(filename: str) -> bool:
    extension = os.path.splitext(filename)[1].lower()
    return extension in ALLOWED_IMAGE_EXTENSIONS


def create_directory(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_upload_file(upload_file: UploadFile, destination_path: str) -> None:
    with open(destination_path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)

import os
import shutil
import time
import uuid
from PIL import Image

import json
from datetime import datetime, timezone

RESULTS_BASE_PATH = "uploads/results"
RESULTS_TTL_SECONDS = 24 * 60 * 60  # 24 horas


def ensure_results_directory() -> (
    None
):  # Crea el directorio base de resultados si no existe
    os.makedirs(RESULTS_BASE_PATH, exist_ok=True)


def cleanup_old_results(ttl_seconds: int = RESULTS_TTL_SECONDS) -> None:
    """
    Elimina directorios de resultados temporales antiguos.
    """
    ensure_results_directory()

    now = time.time()

    for entry in os.listdir(RESULTS_BASE_PATH):
        entry_path = os.path.join(RESULTS_BASE_PATH, entry)

        if not os.path.isdir(entry_path):
            continue

        age_seconds = now - os.path.getmtime(entry_path)

        if age_seconds > ttl_seconds:
            shutil.rmtree(entry_path, ignore_errors=True)


def create_result_directory(prefix: str = "result") -> tuple[str, str]:
    """
    Crea un directorio único para almacenar resultados visuales generados.

    Returns:
        result_id: identificador único del resultado
        result_path: ruta física del directorio
    """
    result_id = f"{prefix}_{uuid.uuid4()}"
    result_path = os.path.join(RESULTS_BASE_PATH, result_id)

    os.makedirs(result_path, exist_ok=True)

    return result_id, result_path


def save_result_image(
    image: Image.Image,
    result_id: str,
    result_path: str,
    filename: str,
) -> str:
    """
    Guarda una imagen PIL como PNG dentro del directorio de resultados
    y devuelve su URL pública.
    """
    image_path = os.path.join(result_path, filename)
    image.save(image_path, format="PNG")

    return f"/results/{result_id}/{filename}"


def save_result_metadata(
    result_path: str,
    metadata: dict,
) -> str:
    metadata_path = os.path.join(result_path, "metadata.json")

    metadata_with_timestamp = {
        **metadata,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata_with_timestamp, file, indent=2)

    return metadata_path

import hashlib
import os
import shutil
import uuid
from pathlib import Path

from PIL import Image

from core.errors import BadRequestError, GoneError
from utils.result_files import RESULTS_BASE_PATH

DETECTION_EIGENCAM_PIPELINE_VERSION = "detection-eigencam-v3-model-contract-spatial"


def detection_evaluation_image_artifact_filenames(
    evaluation_id: str,
    image_id: str,
) -> dict[str, str]:
    identity = f"{evaluation_id}\0{image_id}".encode("utf-8")
    artifact_key = hashlib.sha256(identity).hexdigest()
    eigencam_revision = hashlib.sha256(
        DETECTION_EIGENCAM_PIPELINE_VERSION.encode("utf-8")
    ).hexdigest()[:12]
    return {
        "adversarial_source": f"{artifact_key}_adversarial.png",
        "original_eigencam": (
            f"{artifact_key}_{eigencam_revision}_original_eigencam.png"
        ),
        "adversarial_eigencam": (
            f"{artifact_key}_{eigencam_revision}_adversarial_eigencam.png"
        ),
    }


def detection_evaluation_visual_path(evaluation_id: str) -> Path:
    if os.path.basename(evaluation_id) != evaluation_id:
        raise BadRequestError("La identidad del artefacto de evaluación de detección no es válida")
    return Path(RESULTS_BASE_PATH) / evaluation_id


def save_detection_evaluation_adversarial_image(
    *,
    evaluation_id: str,
    image_id: str,
    image: Image.Image,
) -> str:
    result_path = detection_evaluation_visual_path(evaluation_id)
    result_path.mkdir(parents=True, exist_ok=True)
    filename = detection_evaluation_image_artifact_filenames(
        evaluation_id,
        image_id,
    )["adversarial_source"]
    _save_image_atomically(result_path / filename, image)
    return f"/results/{evaluation_id}/{filename}"


def load_detection_evaluation_adversarial_image(
    *,
    evaluation_id: str,
    image_id: str,
) -> Image.Image:
    result_path = detection_evaluation_visual_path(evaluation_id)
    filename = detection_evaluation_image_artifact_filenames(
        evaluation_id,
        image_id,
    )["adversarial_source"]
    return _load_rgb_image(result_path / filename)


def save_detection_eigencam_pair(
    *,
    evaluation_id: str,
    image_id: str,
    original_eigencam: Image.Image,
    adversarial_eigencam: Image.Image,
) -> None:
    result_path = detection_evaluation_visual_path(evaluation_id)
    result_path.mkdir(parents=True, exist_ok=True)
    filenames = detection_evaluation_image_artifact_filenames(
        evaluation_id,
        image_id,
    )
    _save_image_pair_atomically(
        original_path=result_path / filenames["original_eigencam"],
        adversarial_path=result_path / filenames["adversarial_eigencam"],
        original_image=original_eigencam,
        adversarial_image=adversarial_eigencam,
    )


def detection_eigencam_pair_is_available(
    *,
    evaluation_id: str,
    image_id: str,
) -> bool:
    result_path = detection_evaluation_visual_path(evaluation_id)
    filenames = detection_evaluation_image_artifact_filenames(
        evaluation_id,
        image_id,
    )
    paths = (
        result_path / filenames["original_eigencam"],
        result_path / filenames["adversarial_eigencam"],
    )
    try:
        sizes = []
        for path in paths:
            with Image.open(path) as image:
                image.load()
                sizes.append(image.size)
    except (OSError, ValueError):
        return False
    return sizes[0] == sizes[1]


def remove_detection_evaluation_visual_artifacts(evaluation_id: str) -> None:
    result_path = detection_evaluation_visual_path(evaluation_id)
    if result_path.is_dir():
        shutil.rmtree(result_path)


def _load_rgb_image(path: Path) -> Image.Image:
    if not path.is_file():
        raise GoneError("La imagen tras el ataque de la evaluación de detección ya no está disponible")
    try:
        with Image.open(path) as image:
            return image.copy().convert("RGB")
    except (OSError, ValueError) as error:
        raise BadRequestError(
            "La imagen tras el ataque de la evaluación de detección no es válida"
        ) from error


def _save_image_atomically(path: Path, image: Image.Image) -> None:
    temporary_path = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        image.convert("RGB").save(
            temporary_path,
            format="PNG",
            compress_level=1,
        )
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _save_image_pair_atomically(
    *,
    original_path: Path,
    adversarial_path: Path,
    original_image: Image.Image,
    adversarial_image: Image.Image,
) -> None:
    operation_id = uuid.uuid4().hex
    temporary_original = _transient_path(original_path, operation_id, "tmp")
    temporary_adversarial = _transient_path(adversarial_path, operation_id, "tmp")
    backup_original = _transient_path(original_path, operation_id, "backup")
    backup_adversarial = _transient_path(adversarial_path, operation_id, "backup")
    published_paths: list[Path] = []
    backups: list[tuple[Path, Path]] = []

    try:
        _save_png(original_image, temporary_original)
        _save_png(adversarial_image, temporary_adversarial)
        _backup_existing_paths(
            (
                (original_path, backup_original),
                (adversarial_path, backup_adversarial),
            ),
            backups,
        )
        _publish_path(temporary_original, original_path, published_paths)
        _publish_path(temporary_adversarial, adversarial_path, published_paths)
    except OSError:
        _rollback_published_paths(published_paths, backups)
        raise
    finally:
        _remove_transient_paths(
            temporary_original,
            temporary_adversarial,
            backup_original,
            backup_adversarial,
        )


def _transient_path(path: Path, operation_id: str, suffix: str) -> Path:
    return path.with_name(f"{path.name}.{operation_id}.{suffix}")


def _save_png(image: Image.Image, path: Path) -> None:
    image.convert("RGB").save(path, format="PNG", compress_level=1)


def _backup_existing_paths(
    path_pairs: tuple[tuple[Path, Path], ...],
    backups: list[tuple[Path, Path]],
) -> None:
    for final_path, backup_path in path_pairs:
        if final_path.exists():
            os.replace(final_path, backup_path)
            backups.append((final_path, backup_path))


def _publish_path(
    temporary_path: Path,
    final_path: Path,
    published_paths: list[Path],
) -> None:
    os.replace(temporary_path, final_path)
    published_paths.append(final_path)


def _rollback_published_paths(
    published_paths: list[Path],
    backups: list[tuple[Path, Path]],
) -> None:
    for published_path in reversed(published_paths):
        if published_path.exists():
            published_path.unlink()
    for final_path, backup_path in reversed(backups):
        if backup_path.exists():
            os.replace(backup_path, final_path)


def _remove_transient_paths(*paths: Path) -> None:
    for path in paths:
        if path.exists():
            path.unlink()

import json
import os
import pickle
import threading
import uuid

import torch
from PIL import Image

from core.errors import (
    BadRequestError,
    ForbiddenError,
    GoneError,
    NotFoundError,
)
from core.task_compatibility import (
    CLASSIFICATION_TASK_TYPE,
    require_model_task_type,
)
from ml.explainability.gradcam import generate_gradcam_from_prepared_input
from schemas.evaluation import EvaluationGradCamResponse
from services.model_service import get_active_model_instance
from utils.evaluation_artifacts import (
    load_evaluation_model_inputs,
    validate_evaluation_model_inputs,
)
from utils.result_files import RESULTS_BASE_PATH

_generation_lock = threading.Lock()


async def generate_evaluation_image_gradcam(
    *,
    evaluation_id: str,
    image_id: str,
    current_user_id: str,
) -> EvaluationGradCamResponse:
    result_path = _resolve_evaluation_path(evaluation_id)

    with _generation_lock:
        metadata = _load_metadata(result_path)
        _validate_access_and_state(
            metadata=metadata,
            evaluation_id=evaluation_id,
            current_user_id=current_user_id,
        )
        image_entry = _resolve_image_entry(metadata, image_id)

        cached_response = _cached_response(
            result_path=result_path,
            evaluation_id=evaluation_id,
            image_id=image_id,
            image_entry=image_entry,
        )
        if cached_response is not None:
            return cached_response

        model = get_active_model_instance()
        require_model_task_type(
            model=model,
            expected_task_type=CLASSIFICATION_TASK_TYPE,
            endpoint=(
                f"/evaluations/{evaluation_id}/images/{image_id}/gradcam"
            ),
        )
        if model.id != metadata.get("model_id"):
            raise BadRequestError(
                "The model used by this evaluation is not active"
            )

        clean_model_input, adversarial_model_input = _load_scientific_inputs(
            result_path=result_path,
            evaluation_id=evaluation_id,
            image_id=image_id,
            model=model,
            image_entry=image_entry,
        )
        original_image = _load_rgb_image(
            result_path,
            image_entry.get("original_visual_filename"),
        )
        adversarial_image = _load_rgb_image(
            result_path,
            image_entry.get("adversarial_visual_filename"),
        )

        if original_image.size != adversarial_image.size:
            raise BadRequestError(
                "Evaluation visual artifact geometries do not match"
            )

        clean_model_input = _move_to_model_device(clean_model_input, model)
        adversarial_model_input = _move_to_model_device(
            adversarial_model_input,
            model,
        )

        original_result = generate_gradcam_from_prepared_input(
            model_wrapper=model,
            model_input=clean_model_input,
            visualization_image=original_image,
        )
        adversarial_result = generate_gradcam_from_prepared_input(
            model_wrapper=model,
            model_input=adversarial_model_input,
            visualization_image=adversarial_image,
        )

        original_gradcam = original_result["gradcam_pil_image"]
        adversarial_gradcam = adversarial_result["gradcam_pil_image"]
        if (
            original_gradcam.size != original_image.size
            or adversarial_gradcam.size != original_image.size
        ):
            raise ValueError("Generated Grad-CAM geometry does not match the source")

        _persist_gradcam_pair(
            result_path=result_path,
            original_filename=image_entry["original_gradcam_filename"],
            adversarial_filename=image_entry["adversarial_gradcam_filename"],
            original_gradcam=original_gradcam,
            adversarial_gradcam=adversarial_gradcam,
        )
        image_entry["gradcam_generated"] = True
        _save_metadata_atomically(result_path, metadata)

        response = _build_response(
            evaluation_id=evaluation_id,
            image_id=image_id,
            image_entry=image_entry,
        )

        del clean_model_input
        del adversarial_model_input
        del original_image
        del adversarial_image
        del original_gradcam
        del adversarial_gradcam

        return response


def _resolve_evaluation_path(evaluation_id: str) -> str:
    if (
        not evaluation_id.startswith("evaluation_")
        or os.path.basename(evaluation_id) != evaluation_id
    ):
        raise NotFoundError("Evaluation not found")

    result_path = os.path.join(RESULTS_BASE_PATH, evaluation_id)
    if not os.path.isdir(result_path):
        raise NotFoundError("Evaluation not found")
    return result_path


def _load_metadata(result_path: str) -> dict:
    metadata_path = os.path.join(result_path, "metadata.json")
    if not os.path.isfile(metadata_path):
        raise BadRequestError("Evaluation is not complete")

    try:
        with open(metadata_path, encoding="utf-8") as metadata_file:
            metadata = json.load(metadata_file)
    except (OSError, ValueError) as error:
        raise BadRequestError("Evaluation metadata is invalid") from error

    if not isinstance(metadata, dict):
        raise BadRequestError("Evaluation metadata is invalid")
    return metadata


def _validate_access_and_state(
    *,
    metadata: dict,
    evaluation_id: str,
    current_user_id: str,
) -> None:
    if (
        metadata.get("result_id") != evaluation_id
        or metadata.get("result_type") != "evaluation"
    ):
        raise NotFoundError("Evaluation not found")
    if metadata.get("owner_id") != current_user_id:
        raise ForbiddenError("Evaluation belongs to another user")
    if metadata.get("status") != "completed":
        raise BadRequestError("Evaluation is not complete")


def _resolve_image_entry(metadata: dict, image_id: str) -> dict:
    images = metadata.get("images")
    if not isinstance(images, dict):
        raise BadRequestError("Evaluation image manifest is invalid")
    image_entry = images.get(image_id)
    if not isinstance(image_entry, dict):
        raise NotFoundError("Evaluation image not found")
    return image_entry


def _load_scientific_inputs(
    *,
    result_path: str,
    evaluation_id: str,
    image_id: str,
    model,
    image_entry: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    filename = image_entry.get("model_inputs_filename")
    artifact_path = _resolve_artifact_path(result_path, filename)
    if not os.path.isfile(artifact_path):
        raise GoneError("Evaluation scientific artifact is no longer available")

    try:
        artifact = load_evaluation_model_inputs(artifact_path)
        expected_input_size = _expected_input_size(model)
        return validate_evaluation_model_inputs(
            artifact,
            evaluation_id=evaluation_id,
            image_id=image_id,
            model_id=model.id,
            expected_input_size=expected_input_size,
        )
    except GoneError:
        raise
    except (
        EOFError,
        OSError,
        pickle.UnpicklingError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        raise BadRequestError("Evaluation scientific artifact is invalid") from error


def _load_rgb_image(result_path: str, filename) -> Image.Image:
    image_path = _resolve_artifact_path(result_path, filename)
    if not os.path.isfile(image_path):
        raise GoneError("Evaluation visual artifact is no longer available")
    try:
        with Image.open(image_path) as image:
            return image.copy().convert("RGB")
    except (OSError, ValueError) as error:
        raise BadRequestError("Evaluation visual artifact is invalid") from error


def _resolve_artifact_path(result_path: str, filename) -> str:
    if not isinstance(filename, str) or os.path.basename(filename) != filename:
        raise BadRequestError("Evaluation artifact reference is invalid")
    return os.path.join(result_path, filename)


def _expected_input_size(model) -> tuple[int, int] | None:
    input_size = getattr(model, "input_size", None)
    if not (
        isinstance(input_size, (list, tuple))
        and len(input_size) == 2
        and all(isinstance(value, int) for value in input_size)
    ):
        return None
    return tuple(input_size)


def _move_to_model_device(tensor: torch.Tensor, model) -> torch.Tensor:
    try:
        device = next(model.get_torch_model().parameters()).device
    except (AttributeError, NotImplementedError, StopIteration, TypeError):
        device = getattr(model, "device", torch.device("cpu"))
    return tensor.to(device=device)


def _persist_gradcam_pair(
    *,
    result_path: str,
    original_filename: str,
    adversarial_filename: str,
    original_gradcam: Image.Image,
    adversarial_gradcam: Image.Image,
) -> None:
    original_path = _resolve_artifact_path(result_path, original_filename)
    adversarial_path = _resolve_artifact_path(result_path, adversarial_filename)
    operation_id = uuid.uuid4().hex
    temporary_original = f"{original_path}.{operation_id}.tmp"
    temporary_adversarial = f"{adversarial_path}.{operation_id}.tmp"
    backup_original = f"{original_path}.{operation_id}.backup"
    backup_adversarial = f"{adversarial_path}.{operation_id}.backup"
    published_paths = []
    backup_paths = []

    try:
        original_gradcam.save(temporary_original, format="PNG")
        adversarial_gradcam.save(temporary_adversarial, format="PNG")

        for final_path, backup_path in (
            (original_path, backup_original),
            (adversarial_path, backup_adversarial),
        ):
            if os.path.exists(final_path):
                os.replace(final_path, backup_path)
                backup_paths.append((final_path, backup_path))

        os.replace(temporary_original, original_path)
        published_paths.append(original_path)
        os.replace(temporary_adversarial, adversarial_path)
        published_paths.append(adversarial_path)
    except Exception:
        for published_path in published_paths:
            if os.path.exists(published_path):
                os.remove(published_path)
        for final_path, backup_path in backup_paths:
            if os.path.exists(backup_path):
                os.replace(backup_path, final_path)
        raise
    finally:
        for transient_path in (
            temporary_original,
            temporary_adversarial,
            backup_original,
            backup_adversarial,
        ):
            if os.path.exists(transient_path):
                os.remove(transient_path)


def _save_metadata_atomically(result_path: str, metadata: dict) -> None:
    metadata_path = os.path.join(result_path, "metadata.json")
    temporary_path = f"{metadata_path}.{uuid.uuid4().hex}.tmp"
    try:
        with open(temporary_path, "w", encoding="utf-8") as metadata_file:
            json.dump(metadata, metadata_file, indent=2)
        os.replace(temporary_path, metadata_path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


def _cached_response(
    *,
    result_path: str,
    evaluation_id: str,
    image_id: str,
    image_entry: dict,
) -> EvaluationGradCamResponse | None:
    if image_entry.get("gradcam_generated") is not True:
        return None
    original_path = _resolve_artifact_path(
        result_path,
        image_entry.get("original_gradcam_filename"),
    )
    adversarial_path = _resolve_artifact_path(
        result_path,
        image_entry.get("adversarial_gradcam_filename"),
    )
    if not os.path.isfile(original_path) or not os.path.isfile(adversarial_path):
        return None
    try:
        with Image.open(original_path) as original_image:
            original_image.load()
            original_size = original_image.size
        with Image.open(adversarial_path) as adversarial_image:
            adversarial_image.load()
            adversarial_size = adversarial_image.size
    except (OSError, ValueError):
        return None
    if original_size != adversarial_size:
        return None
    return _build_response(
        evaluation_id=evaluation_id,
        image_id=image_id,
        image_entry=image_entry,
    )


def _build_response(
    *,
    evaluation_id: str,
    image_id: str,
    image_entry: dict,
) -> EvaluationGradCamResponse:
    return EvaluationGradCamResponse(
        evaluation_id=evaluation_id,
        image_id=image_id,
        original_gradcam_image_url=(
            f"/results/{evaluation_id}/"
            f"{image_entry['original_gradcam_filename']}"
        ),
        adversarial_gradcam_image_url=(
            f"/results/{evaluation_id}/"
            f"{image_entry['adversarial_gradcam_filename']}"
        ),
    )

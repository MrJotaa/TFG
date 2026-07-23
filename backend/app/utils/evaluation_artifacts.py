import hashlib
import os
import uuid

import torch

SCIENTIFIC_ARTIFACT_SCHEMA_VERSION = 1


def evaluation_image_artifact_key(
    evaluation_id: str,
    image_id: str,
) -> str:
    identity = f"{evaluation_id}\0{image_id}".encode("utf-8")
    return hashlib.sha256(identity).hexdigest()


def evaluation_image_artifact_filenames(
    evaluation_id: str,
    image_id: str,
) -> dict[str, str]:
    artifact_key = evaluation_image_artifact_key(evaluation_id, image_id)
    return {
        "original_visual": f"{artifact_key}_original.png",
        "model_inputs": f"{artifact_key}_model_inputs.pt",
        "original_gradcam": f"{artifact_key}_original_gradcam.png",
        "adversarial_gradcam": f"{artifact_key}_adversarial_gradcam.png",
    }


def save_evaluation_model_inputs(
    *,
    result_path: str,
    filename: str,
    evaluation_id: str,
    image_id: str,
    model_id: str,
    clean_model_input: torch.Tensor,
    adversarial_model_input: torch.Tensor,
) -> None:
    clean_cpu = _to_persistable_tensor(clean_model_input)
    adversarial_cpu = _to_persistable_tensor(adversarial_model_input)
    artifact_path = os.path.join(result_path, filename)
    temporary_path = f"{artifact_path}.{uuid.uuid4().hex}.tmp"

    try:
        torch.save(
            {
                "schema_version": SCIENTIFIC_ARTIFACT_SCHEMA_VERSION,
                "evaluation_id": evaluation_id,
                "image_id": image_id,
                "model_id": model_id,
                "clean_model_input": clean_cpu,
                "adversarial_model_input": adversarial_cpu,
            },
            temporary_path,
        )
        os.replace(temporary_path, artifact_path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


def load_evaluation_model_inputs(
    artifact_path: str,
) -> dict:
    return torch.load(
        artifact_path,
        map_location="cpu",
        weights_only=True,
    )


def validate_evaluation_model_inputs(
    artifact: dict,
    *,
    evaluation_id: str,
    image_id: str,
    model_id: str,
    expected_input_size: tuple[int, int] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(artifact, dict):
        raise ValueError("Scientific artifact must be a mapping")
    if artifact.get("schema_version") != SCIENTIFIC_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("Unsupported scientific artifact schema")
    if artifact.get("evaluation_id") != evaluation_id:
        raise ValueError("Scientific artifact evaluation does not match")
    if artifact.get("image_id") != image_id:
        raise ValueError("Scientific artifact image does not match")
    if artifact.get("model_id") != model_id:
        raise ValueError("Scientific artifact model does not match")

    clean_model_input = artifact.get("clean_model_input")
    adversarial_model_input = artifact.get("adversarial_model_input")

    _validate_prepared_input(
        clean_model_input,
        expected_input_size=expected_input_size,
        label="Clean",
    )
    _validate_prepared_input(
        adversarial_model_input,
        expected_input_size=expected_input_size,
        label="Adversarial",
    )

    if clean_model_input.shape != adversarial_model_input.shape:
        raise ValueError("Scientific input shapes do not match")
    if clean_model_input.dtype != adversarial_model_input.dtype:
        raise ValueError("Scientific input dtypes do not match")

    return clean_model_input, adversarial_model_input


def _to_persistable_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Scientific model input must be a torch.Tensor")
    return tensor.detach().to(device="cpu", copy=True)


def _validate_prepared_input(
    tensor,
    *,
    expected_input_size: tuple[int, int] | None,
    label: str,
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"{label} model input must be a tensor")
    if tensor.ndim != 4 or tensor.shape[0] != 1 or tensor.shape[1] != 3:
        raise ValueError(f"{label} model input must have shape [1, 3, H, W]")
    if not tensor.is_floating_point():
        raise ValueError(f"{label} model input must be floating point")
    if tensor.device.type != "cpu":
        raise ValueError(f"{label} model input must be stored on CPU")
    if tensor.requires_grad or tensor.grad_fn is not None:
        raise ValueError(f"{label} model input must be detached")
    if not torch.isfinite(tensor).all().item():
        raise ValueError(f"{label} model input contains non-finite values")
    if tensor.numel() and (tensor.min().item() < 0.0 or tensor.max().item() > 1.0):
        raise ValueError(f"{label} model input must be in the [0, 1] domain")

    if expected_input_size is not None:
        expected_height, expected_width = expected_input_size
        if tensor.shape[-2:] != (expected_height, expected_width):
            raise ValueError(f"{label} model input geometry does not match the model")

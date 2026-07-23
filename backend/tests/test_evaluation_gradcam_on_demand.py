import asyncio
import json
import os
import time
from pathlib import Path

import pytest
import torch
from PIL import Image

import api.routes.evaluate as evaluate_route
import services.evaluation_gradcam_service as gradcam_service
import utils.result_files as result_files
from api.dependencies.auth import require_current_user
from core.errors import (
    BadRequestError,
    ForbiddenError,
    GoneError,
    NotFoundError,
)
from schemas.evaluation import EvaluationGradCamResponse
from utils.evaluation_artifacts import (
    evaluation_image_artifact_filenames,
    load_evaluation_model_inputs,
    save_evaluation_model_inputs,
    validate_evaluation_model_inputs,
)


class _EvaluationModel:
    id = "test-model"
    task_type = "classification"
    input_size = (5, 7)
    device = torch.device("cpu")

    def get_torch_model(self):
        raise NotImplementedError


def _write_evaluation(
    base_path: Path,
    *,
    evaluation_id: str = "evaluation_test",
    image_id: str = "image-1",
    owner_id: str = "user-1",
    source_size: tuple[int, int] = (19, 11),
    status: str = "completed",
    model_id: str = "test-model",
) -> tuple[Path, dict]:
    result_path = base_path / evaluation_id
    result_path.mkdir()
    filenames = evaluation_image_artifact_filenames(evaluation_id, image_id)
    adversarial_filename = f"{image_id}_adversarial.png"

    Image.new("RGB", source_size, color=(10, 20, 30)).save(
        result_path / filenames["original_visual"],
        format="PNG",
    )
    Image.new("RGB", source_size, color=(40, 50, 60)).save(
        result_path / adversarial_filename,
        format="PNG",
    )

    clean_model_input = torch.full((1, 3, 5, 7), 0.25)
    adversarial_model_input = torch.full((1, 3, 5, 7), 0.5)
    save_evaluation_model_inputs(
        result_path=str(result_path),
        filename=filenames["model_inputs"],
        evaluation_id=evaluation_id,
        image_id=image_id,
        model_id=model_id,
        clean_model_input=clean_model_input,
        adversarial_model_input=adversarial_model_input,
    )

    image_entry = {
        "image_id": image_id,
        "filename": "shared-name.png",
        "original_visual_filename": filenames["original_visual"],
        "adversarial_visual_filename": adversarial_filename,
        "model_inputs_filename": filenames["model_inputs"],
        "original_gradcam_filename": filenames["original_gradcam"],
        "adversarial_gradcam_filename": filenames["adversarial_gradcam"],
        "gradcam_generated": False,
    }
    metadata = {
        "result_id": evaluation_id,
        "result_type": "evaluation",
        "status": status,
        "owner_id": owner_id,
        "model_id": model_id,
        "images": {image_id: image_entry},
    }
    (result_path / "metadata.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )
    return result_path, image_entry


@pytest.mark.parametrize(
    "source_size",
    [
        (31, 11),
        (13, 37),
        (23, 23),
        (19, 10),
    ],
)
@pytest.mark.asyncio
async def test_on_demand_gradcam_uses_stored_tensors_and_source_geometry(
    monkeypatch,
    tmp_path,
    source_size,
):
    evaluation_id = f"evaluation_{source_size[0]}_{source_size[1]}"
    image_id = "image-1"
    result_path, image_entry = _write_evaluation(
        tmp_path,
        evaluation_id=evaluation_id,
        image_id=image_id,
        source_size=source_size,
    )
    generator_calls = []
    scientific_loads = 0
    real_load = gradcam_service.load_evaluation_model_inputs

    def fake_generate_gradcam_from_prepared_input(
        model_wrapper,
        model_input,
        visualization_image,
    ):
        rendered = visualization_image.copy()
        generator_calls.append(
            {
                "model": model_wrapper,
                "model_input": model_input,
                "visualization_image": visualization_image,
            }
        )
        return {"gradcam_pil_image": rendered}

    def tracked_load(artifact_path):
        nonlocal scientific_loads
        scientific_loads += 1
        return real_load(artifact_path)

    monkeypatch.setattr(gradcam_service, "RESULTS_BASE_PATH", str(tmp_path))
    monkeypatch.setattr(
        gradcam_service,
        "get_active_model_instance",
        lambda: _EvaluationModel(),
    )
    monkeypatch.setattr(
        gradcam_service,
        "generate_gradcam_from_prepared_input",
        fake_generate_gradcam_from_prepared_input,
    )
    monkeypatch.setattr(
        gradcam_service,
        "load_evaluation_model_inputs",
        tracked_load,
    )

    first_response = await gradcam_service.generate_evaluation_image_gradcam(
        evaluation_id=evaluation_id,
        image_id=image_id,
        current_user_id="user-1",
    )
    second_response = await gradcam_service.generate_evaluation_image_gradcam(
        evaluation_id=evaluation_id,
        image_id=image_id,
        current_user_id="user-1",
    )

    assert first_response == second_response
    assert len(generator_calls) == 2
    assert scientific_loads == 1
    assert torch.equal(
        generator_calls[0]["model_input"],
        torch.full((1, 3, 5, 7), 0.25),
    )
    assert torch.equal(
        generator_calls[1]["model_input"],
        torch.full((1, 3, 5, 7), 0.5),
    )
    assert generator_calls[0]["visualization_image"].size == source_size
    assert generator_calls[0]["visualization_image"].getpixel((0, 0)) == (
        10,
        20,
        30,
    )
    assert generator_calls[1]["visualization_image"].size == source_size
    assert generator_calls[1]["visualization_image"].getpixel((0, 0)) == (
        40,
        50,
        60,
    )

    for filename in (
        image_entry["original_gradcam_filename"],
        image_entry["adversarial_gradcam_filename"],
    ):
        with Image.open(result_path / filename) as gradcam_image:
            assert gradcam_image.size == source_size
            assert gradcam_image.mode == "RGB"

    updated_metadata = json.loads(
        (result_path / "metadata.json").read_text(encoding="utf-8")
    )
    assert updated_metadata["images"][image_id]["gradcam_generated"] is True


@pytest.mark.asyncio
async def test_on_demand_gradcam_passes_loaded_tensors_by_identity(
    monkeypatch,
    tmp_path,
):
    evaluation_id = "evaluation_identity"
    image_id = "image-1"
    _write_evaluation(
        tmp_path,
        evaluation_id=evaluation_id,
        image_id=image_id,
    )
    clean_model_input = torch.full((1, 3, 5, 7), 0.2)
    adversarial_model_input = torch.full((1, 3, 5, 7), 0.7)
    artifact = {
        "schema_version": 1,
        "evaluation_id": evaluation_id,
        "image_id": image_id,
        "model_id": "test-model",
        "clean_model_input": clean_model_input,
        "adversarial_model_input": adversarial_model_input,
    }
    received_inputs = []

    monkeypatch.setattr(gradcam_service, "RESULTS_BASE_PATH", str(tmp_path))
    monkeypatch.setattr(
        gradcam_service,
        "get_active_model_instance",
        lambda: _EvaluationModel(),
    )
    monkeypatch.setattr(
        gradcam_service,
        "load_evaluation_model_inputs",
        lambda artifact_path: artifact,
    )

    def fake_generate(model_wrapper, model_input, visualization_image):
        received_inputs.append(model_input)
        return {"gradcam_pil_image": visualization_image.copy()}

    monkeypatch.setattr(
        gradcam_service,
        "generate_gradcam_from_prepared_input",
        fake_generate,
    )

    await gradcam_service.generate_evaluation_image_gradcam(
        evaluation_id=evaluation_id,
        image_id=image_id,
        current_user_id="user-1",
    )

    assert received_inputs == [clean_model_input, adversarial_model_input]


@pytest.mark.parametrize(
    ("scenario", "exception", "message"),
    [
        ("foreign", ForbiddenError, "another user"),
        ("missing_evaluation", NotFoundError, "Evaluation not found"),
        ("missing_image", NotFoundError, "Evaluation image not found"),
        ("processing", BadRequestError, "not complete"),
        ("missing_tensor", GoneError, "scientific artifact"),
        ("missing_visual", GoneError, "visual artifact"),
        ("wrong_model", BadRequestError, "not active"),
    ],
)
@pytest.mark.asyncio
async def test_on_demand_gradcam_rejects_invalid_access_or_state(
    monkeypatch,
    tmp_path,
    scenario,
    exception,
    message,
):
    evaluation_id = "evaluation_errors"
    image_id = "image-1"
    result_path, image_entry = _write_evaluation(
        tmp_path,
        evaluation_id=evaluation_id,
        image_id=image_id,
        status="processing" if scenario == "processing" else "completed",
        model_id="other-model" if scenario == "wrong_model" else "test-model",
    )
    requested_evaluation_id = evaluation_id
    requested_image_id = image_id
    current_user_id = "user-2" if scenario == "foreign" else "user-1"

    if scenario == "missing_evaluation":
        requested_evaluation_id = "evaluation_missing"
    elif scenario == "missing_image":
        requested_image_id = "image-missing"
    elif scenario == "missing_tensor":
        (result_path / image_entry["model_inputs_filename"]).unlink()
    elif scenario == "missing_visual":
        (result_path / image_entry["original_visual_filename"]).unlink()

    monkeypatch.setattr(gradcam_service, "RESULTS_BASE_PATH", str(tmp_path))
    monkeypatch.setattr(
        gradcam_service,
        "get_active_model_instance",
        lambda: _EvaluationModel(),
    )

    with pytest.raises(exception, match=message):
        await gradcam_service.generate_evaluation_image_gradcam(
            evaluation_id=requested_evaluation_id,
            image_id=requested_image_id,
            current_user_id=current_user_id,
        )


@pytest.mark.parametrize("corruption", ["shape", "non_finite", "wrong_identity"])
@pytest.mark.asyncio
async def test_on_demand_gradcam_rejects_corrupt_scientific_artifacts(
    monkeypatch,
    tmp_path,
    corruption,
):
    evaluation_id = f"evaluation_corrupt_{corruption}"
    image_id = "image-1"
    _write_evaluation(
        tmp_path,
        evaluation_id=evaluation_id,
        image_id=image_id,
    )
    clean = torch.zeros((1, 3, 5, 7))
    adversarial = torch.ones((1, 3, 5, 7))
    if corruption == "shape":
        adversarial = torch.ones((1, 1, 5, 7))
    elif corruption == "non_finite":
        adversarial[0, 0, 0, 0] = torch.nan

    artifact = {
        "schema_version": 1,
        "evaluation_id": (
            "evaluation_other" if corruption == "wrong_identity" else evaluation_id
        ),
        "image_id": image_id,
        "model_id": "test-model",
        "clean_model_input": clean,
        "adversarial_model_input": adversarial,
    }

    monkeypatch.setattr(gradcam_service, "RESULTS_BASE_PATH", str(tmp_path))
    monkeypatch.setattr(
        gradcam_service,
        "get_active_model_instance",
        lambda: _EvaluationModel(),
    )
    monkeypatch.setattr(
        gradcam_service,
        "load_evaluation_model_inputs",
        lambda artifact_path: artifact,
    )

    with pytest.raises(BadRequestError, match="scientific artifact is invalid"):
        await gradcam_service.generate_evaluation_image_gradcam(
            evaluation_id=evaluation_id,
            image_id=image_id,
            current_user_id="user-1",
        )


@pytest.mark.parametrize("failed_call", [1, 2])
@pytest.mark.asyncio
async def test_gradcam_failure_does_not_publish_or_mark_partial_pair(
    monkeypatch,
    tmp_path,
    failed_call,
):
    evaluation_id = "evaluation_partial"
    image_id = "image-1"
    result_path, image_entry = _write_evaluation(
        tmp_path,
        evaluation_id=evaluation_id,
        image_id=image_id,
    )
    calls = 0

    def failing_gradcam(model_wrapper, model_input, visualization_image):
        nonlocal calls
        calls += 1
        if calls == failed_call:
            raise RuntimeError("Grad-CAM failed")
        return {"gradcam_pil_image": visualization_image.copy()}

    monkeypatch.setattr(gradcam_service, "RESULTS_BASE_PATH", str(tmp_path))
    monkeypatch.setattr(
        gradcam_service,
        "get_active_model_instance",
        lambda: _EvaluationModel(),
    )
    monkeypatch.setattr(
        gradcam_service,
        "generate_gradcam_from_prepared_input",
        failing_gradcam,
    )

    with pytest.raises(RuntimeError, match="Grad-CAM failed"):
        await gradcam_service.generate_evaluation_image_gradcam(
            evaluation_id=evaluation_id,
            image_id=image_id,
            current_user_id="user-1",
        )

    assert not (result_path / image_entry["original_gradcam_filename"]).exists()
    assert not (result_path / image_entry["adversarial_gradcam_filename"]).exists()
    metadata = json.loads((result_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["images"][image_id]["gradcam_generated"] is False


@pytest.mark.asyncio
async def test_persistence_failure_does_not_mark_gradcam_as_generated(
    monkeypatch,
    tmp_path,
):
    evaluation_id = "evaluation_persistence_failure"
    image_id = "image-1"
    result_path, image_entry = _write_evaluation(
        tmp_path,
        evaluation_id=evaluation_id,
        image_id=image_id,
    )

    monkeypatch.setattr(gradcam_service, "RESULTS_BASE_PATH", str(tmp_path))
    monkeypatch.setattr(
        gradcam_service,
        "get_active_model_instance",
        lambda: _EvaluationModel(),
    )
    monkeypatch.setattr(
        gradcam_service,
        "generate_gradcam_from_prepared_input",
        lambda model_wrapper, model_input, visualization_image: {
            "gradcam_pil_image": visualization_image.copy()
        },
    )
    monkeypatch.setattr(
        gradcam_service,
        "_persist_gradcam_pair",
        lambda **kwargs: (_ for _ in ()).throw(OSError("persistence failed")),
    )

    with pytest.raises(OSError, match="persistence failed"):
        await gradcam_service.generate_evaluation_image_gradcam(
            evaluation_id=evaluation_id,
            image_id=image_id,
            current_user_id="user-1",
        )

    assert not (result_path / image_entry["original_gradcam_filename"]).exists()
    assert not (result_path / image_entry["adversarial_gradcam_filename"]).exists()
    metadata = json.loads((result_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["images"][image_id]["gradcam_generated"] is False


def test_pair_publication_rolls_back_if_second_publish_fails(
    monkeypatch,
    tmp_path,
):
    original_filename = "original_gradcam.png"
    adversarial_filename = "adversarial_gradcam.png"
    original_path = tmp_path / original_filename
    adversarial_path = tmp_path / adversarial_filename
    Image.new("RGB", (5, 3), color=(1, 2, 3)).save(original_path)
    Image.new("RGB", (5, 3), color=(4, 5, 6)).save(adversarial_path)
    real_replace = gradcam_service.os.replace

    def fail_second_publish(source, destination):
        if str(source).endswith(".tmp") and destination == str(adversarial_path):
            raise OSError("second publish failed")
        real_replace(source, destination)

    monkeypatch.setattr(gradcam_service.os, "replace", fail_second_publish)

    with pytest.raises(OSError, match="second publish failed"):
        gradcam_service._persist_gradcam_pair(
            result_path=str(tmp_path),
            original_filename=original_filename,
            adversarial_filename=adversarial_filename,
            original_gradcam=Image.new("RGB", (5, 3), color=(10, 20, 30)),
            adversarial_gradcam=Image.new(
                "RGB",
                (5, 3),
                color=(40, 50, 60),
            ),
        )

    with Image.open(original_path) as original:
        assert original.getpixel((0, 0)) == (1, 2, 3)
    with Image.open(adversarial_path) as adversarial:
        assert adversarial.getpixel((0, 0)) == (4, 5, 6)
    assert not list(tmp_path.glob("*.tmp"))
    assert not list(tmp_path.glob("*.backup"))


@pytest.mark.asyncio
async def test_concurrent_requests_publish_one_associated_pair(
    monkeypatch,
    tmp_path,
):
    evaluation_id = "evaluation_concurrent"
    image_id = "image-1"
    _write_evaluation(
        tmp_path,
        evaluation_id=evaluation_id,
        image_id=image_id,
    )
    calls = 0

    def tracked_generate(model_wrapper, model_input, visualization_image):
        nonlocal calls
        calls += 1
        return {"gradcam_pil_image": visualization_image.copy()}

    monkeypatch.setattr(gradcam_service, "RESULTS_BASE_PATH", str(tmp_path))
    monkeypatch.setattr(
        gradcam_service,
        "get_active_model_instance",
        lambda: _EvaluationModel(),
    )
    monkeypatch.setattr(
        gradcam_service,
        "generate_gradcam_from_prepared_input",
        tracked_generate,
    )

    first, second = await asyncio.gather(
        gradcam_service.generate_evaluation_image_gradcam(
            evaluation_id=evaluation_id,
            image_id=image_id,
            current_user_id="user-1",
        ),
        gradcam_service.generate_evaluation_image_gradcam(
            evaluation_id=evaluation_id,
            image_id=image_id,
            current_user_id="user-1",
        ),
    )

    assert first == second
    assert calls == 2


def test_scientific_artifact_is_cpu_detached_and_validated(tmp_path):
    source_clean = torch.full(
        (1, 3, 5, 7),
        0.25,
        requires_grad=True,
    )
    source_adversarial = (source_clean.detach() + 0.1).requires_grad_(True)
    filename = "inputs.pt"

    save_evaluation_model_inputs(
        result_path=str(tmp_path),
        filename=filename,
        evaluation_id="evaluation_artifact",
        image_id="image-1",
        model_id="test-model",
        clean_model_input=source_clean,
        adversarial_model_input=source_adversarial,
    )
    artifact = load_evaluation_model_inputs(str(tmp_path / filename))
    clean, adversarial = validate_evaluation_model_inputs(
        artifact,
        evaluation_id="evaluation_artifact",
        image_id="image-1",
        model_id="test-model",
        expected_input_size=(5, 7),
    )

    assert clean.device.type == "cpu"
    assert adversarial.device.type == "cpu"
    assert clean.requires_grad is False
    assert adversarial.requires_grad is False
    assert clean.grad_fn is None
    assert adversarial.grad_fn is None
    assert torch.equal(clean, source_clean.detach())
    assert torch.equal(adversarial, source_adversarial.detach())


def test_artifact_keys_separate_same_filename_images():
    first = evaluation_image_artifact_filenames(
        "evaluation_one",
        "folder-a/shared.png",
    )
    second = evaluation_image_artifact_filenames(
        "evaluation_one",
        "folder-b/shared.png",
    )
    other_evaluation = evaluation_image_artifact_filenames(
        "evaluation_two",
        "folder-a/shared.png",
    )

    assert first != second
    assert first != other_evaluation


def test_evaluation_cleanup_removes_visual_scientific_and_gradcam_artifacts(
    monkeypatch,
    tmp_path,
):
    result_path, image_entry = _write_evaluation(
        tmp_path,
        evaluation_id="evaluation_cleanup",
    )
    Image.new("RGB", (19, 11)).save(
        result_path / image_entry["original_gradcam_filename"],
        format="PNG",
    )
    Image.new("RGB", (19, 11)).save(
        result_path / image_entry["adversarial_gradcam_filename"],
        format="PNG",
    )
    old_timestamp = time.time() - 60
    os.utime(result_path, (old_timestamp, old_timestamp))
    monkeypatch.setattr(result_files, "RESULTS_BASE_PATH", str(tmp_path))

    result_files.cleanup_old_results(ttl_seconds=1)

    assert not result_path.exists()


def test_evaluation_gradcam_route_is_authenticated_post():
    route = next(
        route
        for route in evaluate_route.router.routes
        if route.path == "/evaluations/{evaluation_id}/images/{image_id}/gradcam"
    )

    assert route.methods == {"POST"}
    assert route.response_model is EvaluationGradCamResponse
    assert any(
        dependency.call is require_current_user
        for dependency in route.dependant.dependencies
    )

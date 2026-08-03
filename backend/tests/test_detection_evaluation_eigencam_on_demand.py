from types import SimpleNamespace

import pytest
from PIL import Image

import services.detection_evaluation_explainability_service as eigencam_service
import utils.detection_evaluation_artifacts as artifacts
from core.errors import BadRequestError, GoneError
from schemas.detection import BoundingBox, DetectionPrediction
from schemas.detection_evaluation import DetectionEvaluationImageResult

EVALUATION_ID = "detection_evaluation_00000000-0000-4000-8000-000000000071"
IMAGE_ID = "image-1"


def _image_result(*, adversarial_image_url: str | None = "/results/adversarial.png"):
    detection = DetectionPrediction(
        label="car",
        confidence=0.82,
        bbox=BoundingBox(x1=2, y1=3, x2=12, y2=14),
    )
    return DetectionEvaluationImageResult(
        image_id=IMAGE_ID,
        filename="image.png",
        original_detection_count=1,
        adversarial_detection_count=1,
        detections_difference=0,
        average_original_confidence=0.9,
        average_adversarial_confidence=0.82,
        average_confidence_difference=0.08,
        attack_success=True,
        perturbation_linf=0.03,
        perturbation_l2=0.5,
        adversarial_image_url=adversarial_image_url,
        original_detections=[detection],
        adversarial_detections=[detection],
    )


@pytest.mark.asyncio
async def test_detection_eigencam_reuses_evaluation_inputs_and_caches(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(artifacts, "RESULTS_BASE_PATH", str(tmp_path))
    artifacts.save_detection_evaluation_adversarial_image(
        evaluation_id=EVALUATION_ID,
        image_id=IMAGE_ID,
        image=Image.new("RGB", (23, 17), color=(40, 50, 60)),
    )

    evaluation = SimpleNamespace(
        dataset_id="dataset-1",
        model_id="detector-1",
        results=[_image_result()],
    )
    model = SimpleNamespace(id="detector-1", task_type="detection")
    generated_from = []
    original_loads = 0

    async def fake_get_evaluation(*, evaluation_id, owner_id):
        assert evaluation_id == EVALUATION_ID
        assert owner_id == "user-1"
        return evaluation

    async def fake_get_image(*, dataset_id, image_id):
        nonlocal original_loads
        original_loads += 1
        assert dataset_id == "dataset-1"
        assert image_id == IMAGE_ID
        return Image.new("RGB", (23, 17), color=(10, 20, 30))

    def fake_generate(received_model, image):
        assert received_model is model
        generated_from.append(image.getpixel((0, 0)))
        return image.copy()

    monkeypatch.setattr(
        eigencam_service,
        "get_detection_evaluation",
        fake_get_evaluation,
    )
    monkeypatch.setattr(
        eigencam_service,
        "get_image_from_dataset",
        fake_get_image,
    )
    monkeypatch.setattr(
        eigencam_service,
        "get_active_model_instance",
        lambda: model,
    )
    monkeypatch.setattr(
        eigencam_service,
        "generate_detection_eigencam",
        fake_generate,
    )

    first = await eigencam_service.generate_detection_evaluation_image_eigencam(
        evaluation_id=EVALUATION_ID,
        image_id=IMAGE_ID,
        current_user_id="user-1",
    )
    second = await eigencam_service.generate_detection_evaluation_image_eigencam(
        evaluation_id=EVALUATION_ID,
        image_id=IMAGE_ID,
        current_user_id="user-1",
    )

    assert first == second
    assert first.explanation_type == "eigencam"
    assert generated_from == [(10, 20, 30), (40, 50, 60)]
    assert original_loads == 1

    filenames = artifacts.detection_evaluation_image_artifact_filenames(
        EVALUATION_ID,
        IMAGE_ID,
    )
    for key in ("original_eigencam", "adversarial_eigencam"):
        with Image.open(tmp_path / EVALUATION_ID / filenames[key]) as image:
            assert image.size == (23, 17)
            assert image.mode == "RGB"


@pytest.mark.asyncio
async def test_detection_eigencam_requires_persisted_adversarial_source(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(artifacts, "RESULTS_BASE_PATH", str(tmp_path))
    evaluation = SimpleNamespace(
        dataset_id="dataset-1",
        model_id="detector-1",
        results=[_image_result(adversarial_image_url=None)],
    )

    async def fake_get_evaluation(**kwargs):
        return evaluation

    monkeypatch.setattr(
        eigencam_service,
        "get_detection_evaluation",
        fake_get_evaluation,
    )

    with pytest.raises(GoneError, match="repeat the evaluation"):
        await eigencam_service.generate_detection_evaluation_image_eigencam(
            evaluation_id=EVALUATION_ID,
            image_id=IMAGE_ID,
            current_user_id="user-1",
        )


@pytest.mark.asyncio
async def test_detection_eigencam_requires_the_evaluation_model(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(artifacts, "RESULTS_BASE_PATH", str(tmp_path))
    artifacts.save_detection_evaluation_adversarial_image(
        evaluation_id=EVALUATION_ID,
        image_id=IMAGE_ID,
        image=Image.new("RGB", (23, 17)),
    )
    evaluation = SimpleNamespace(
        dataset_id="dataset-1",
        model_id="detector-1",
        results=[_image_result()],
    )

    async def fake_get_evaluation(**kwargs):
        return evaluation

    monkeypatch.setattr(
        eigencam_service,
        "get_detection_evaluation",
        fake_get_evaluation,
    )
    monkeypatch.setattr(
        eigencam_service,
        "get_active_model_instance",
        lambda: SimpleNamespace(id="detector-2", task_type="detection"),
    )

    with pytest.raises(BadRequestError, match="not active"):
        await eigencam_service.generate_detection_evaluation_image_eigencam(
            evaluation_id=EVALUATION_ID,
            image_id=IMAGE_ID,
            current_user_id="user-1",
        )

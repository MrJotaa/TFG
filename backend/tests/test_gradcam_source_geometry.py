from io import BytesIO

import numpy as np
import pytest
import torch
from PIL import Image

import api.routes.explainability as explainability_route
import ml.explainability.gradcam as gradcam_module
import services.explainability_service as explainability_service
from main import app
from schemas.explainability import GradCamRequest, GradCamResponse


class _TrackingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.target_layer = torch.nn.Conv2d(3, 2, kernel_size=1, bias=False)
        self.forward_count = 0
        self.received_input = None

        with torch.no_grad():
            self.target_layer.weight.copy_(
                torch.tensor(
                    [
                        [[[1.0]], [[0.0]], [[0.0]]],
                        [[[0.0]], [[1.0]], [[0.0]]],
                    ]
                )
            )

    def forward(self, model_input):
        self.forward_count += 1
        self.received_input = model_input
        return self.target_layer(model_input).mean(dim=(2, 3))


class _TrackingWrapper:
    categories = ("red", "green")

    def __init__(self):
        self.model = _TrackingModel()
        self.preprocess_calls = 0

    def get_torch_model(self):
        return self.model

    def get_gradcam_target_layer(self):
        return self.model.target_layer

    def preprocess(self, image):
        self.preprocess_calls += 1
        return torch.full((1, 3, 5, 7), 0.25)


def _constant_red_colormap(values):
    rgba = np.zeros((*values.shape, 4), dtype=np.float64)
    rgba[..., 0] = 1.0
    rgba[..., 3] = 1.0
    return rgba


@pytest.mark.parametrize(
    "source_size",
    [
        (31, 11),
        (13, 37),
        (23, 23),
        (19, 10),
    ],
)
def test_clean_gradcam_preserves_source_geometry_and_visual_semantics(
    monkeypatch,
    source_size,
):
    wrapper = _TrackingWrapper()
    model_input = torch.zeros((1, 3, 5, 7))
    model_input[:, 0] = 0.8
    source_image = Image.new("RGB", source_size, color=(20, 40, 60))
    source_before = source_image.tobytes()
    target_layer = wrapper.get_gradcam_target_layer()

    monkeypatch.setattr(gradcam_module.cm, "jet", _constant_red_colormap)

    result = gradcam_module.generate_clean_gradcam(
        model_wrapper=wrapper,
        model_input=model_input,
        source_image=source_image,
    )
    persisted_png = BytesIO()
    result["gradcam_pil_image"].save(persisted_png, format="PNG")
    persisted_png.seek(0)

    with Image.open(persisted_png) as reopened_png:
        assert reopened_png.size == source_size
        assert reopened_png.mode == "RGB"

    expected_pixel = tuple(
        np.uint8(
            0.45 * np.array([255, 0, 0])
            + 0.55 * np.array([20, 40, 60])
        )
    )

    assert result["gradcam_pil_image"].size == source_size
    assert result["gradcam_pil_image"].mode == "RGB"
    assert result["gradcam_pil_image"].getpixel((0, 0)) == expected_pixel
    assert wrapper.model.received_input is model_input
    assert wrapper.model.forward_count == 1
    assert wrapper.preprocess_calls == 0
    assert source_image.size == source_size
    assert source_image.tobytes() == source_before
    assert len(target_layer._forward_hooks) == 0
    assert len(target_layer._backward_hooks) == 0
    assert all(parameter.grad is None for parameter in wrapper.model.parameters())


def test_clean_gradcam_removes_hooks_when_forward_fails(monkeypatch):
    wrapper = _TrackingWrapper()
    target_layer = wrapper.get_gradcam_target_layer()

    def failing_forward(model_input):
        target_layer(model_input)
        raise RuntimeError("forward failed")

    monkeypatch.setattr(wrapper.model, "forward", failing_forward)

    with pytest.raises(RuntimeError, match="forward failed"):
        gradcam_module.generate_clean_gradcam(
            model_wrapper=wrapper,
            model_input=torch.zeros((1, 3, 5, 7)),
            source_image=Image.new("RGB", (19, 11)),
        )

    assert len(target_layer._forward_hooks) == 0
    assert len(target_layer._backward_hooks) == 0


def test_legacy_gradcam_consumer_keeps_preprocessing_contract():
    wrapper = _TrackingWrapper()
    image = Image.new("RGB", (17, 9), color=(80, 90, 100))

    result = gradcam_module.generate_gradcam(wrapper, image)

    assert wrapper.preprocess_calls == 1
    assert wrapper.model.received_input.shape == (1, 3, 5, 7)
    assert wrapper.model.forward_count == 1
    assert result["gradcam_pil_image"].size == image.size
    assert result["gradcam_pil_image"].mode == "RGB"


@pytest.mark.asyncio
async def test_explain_endpoint_service_separates_model_input_and_source(
    monkeypatch,
):
    source_image = Image.new("RGB", (29, 13), color=(12, 34, 56))
    model_input = object()
    captured = {}
    saved = {}

    class _ServiceModel:
        id = "classification-model"
        task_type = "classification"

        def preprocess(self, image):
            assert image is source_image
            captured["preprocess_calls"] = (
                captured.get("preprocess_calls", 0) + 1
            )
            return model_input

    async def fake_get_image_from_dataset(dataset_id, image_id):
        assert dataset_id == "dataset-1"
        assert image_id == "image-1"
        return source_image

    def fake_generate_clean_gradcam(
        model_wrapper,
        model_input,
        source_image,
    ):
        captured["model_wrapper"] = model_wrapper
        captured["model_input"] = model_input
        captured["source_image"] = source_image
        return {
            "prediction": "class-a",
            "confidence": 0.75,
            "gradcam_pil_image": source_image.copy(),
        }

    def fake_save_result_image(image, result_id, result_path, filename):
        saved["image"] = image
        saved["filename"] = filename
        return f"/results/{result_id}/{filename}"

    model = _ServiceModel()
    monkeypatch.setattr(
        explainability_service,
        "get_image_from_dataset",
        fake_get_image_from_dataset,
    )
    monkeypatch.setattr(
        explainability_service,
        "get_active_model_instance",
        lambda: model,
    )
    monkeypatch.setattr(
        explainability_service,
        "generate_clean_gradcam",
        fake_generate_clean_gradcam,
    )
    monkeypatch.setattr(
        explainability_service,
        "create_result_directory",
        lambda prefix: ("gradcam-result", "/tmp/gradcam-result"),
    )
    monkeypatch.setattr(
        explainability_service,
        "save_result_image",
        fake_save_result_image,
    )
    monkeypatch.setattr(
        explainability_service,
        "save_result_metadata",
        lambda result_path, metadata: None,
    )

    response = await explainability_service.generate_gradcam_for_image(
        dataset_id="dataset-1",
        image_id="image-1",
    )

    assert captured["preprocess_calls"] == 1
    assert captured["model_wrapper"] is model
    assert captured["model_input"] is model_input
    assert captured["source_image"] is not source_image
    assert captured["source_image"].mode == "RGB"
    assert captured["source_image"].size == source_image.size
    assert captured["source_image"].tobytes() == source_image.tobytes()
    assert saved["image"].size == source_image.size
    assert saved["filename"] == "gradcam.png"
    assert response.model_dump() == {
        "result_id": "gradcam-result",
        "prediction": "class-a",
        "confidence": 0.75,
        "gradcam_image_url": "/results/gradcam-result/gradcam.png",
    }


@pytest.mark.asyncio
async def test_gradcam_http_contract_remains_unchanged(monkeypatch):
    async def fake_generate_gradcam_for_image(dataset_id, image_id):
        assert dataset_id == "dataset-1"
        assert image_id == "image-1"
        return {
            "result_id": "gradcam-result",
            "prediction": "class-a",
            "confidence": 0.75,
            "gradcam_image_url": "/results/gradcam-result/gradcam.png",
        }

    monkeypatch.setattr(
        explainability_route,
        "generate_gradcam_for_image",
        fake_generate_gradcam_for_image,
    )

    gradcam_route = next(
        route
        for route in app.routes
        if getattr(route, "path", None) == "/explain/gradcam"
    )
    assert gradcam_route.methods == {"POST"}
    assert gradcam_route.response_model is GradCamResponse

    response = await explainability_route.gradcam(
        GradCamRequest(
            dataset_id="dataset-1",
            image_id="image-1",
        )
    )

    assert response == {
        "result_id": "gradcam-result",
        "prediction": "class-a",
        "confidence": 0.75,
        "gradcam_image_url": "/results/gradcam-result/gradcam.png",
    }

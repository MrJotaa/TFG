from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from ultralytics.engine.results import Results

import ml.models.ultralytics.adapter as ultralytics_adapter
import services.detection_attack_service as detection_attack_service
import services.detection_service as detection_service
import services.explainability_service as explainability_service
from ml.models.detection import Detection, DetectionBox, DetectionResult
from ml.models.detection_inference import (
    DetectionEvaluationInferenceConfiguration,
)
from ml.models.ultralytics.adapter import (
    DetectionAdapterError,
    UltralyticsDetectionAdapter,
)
from ml.models.ultralytics.config import UltralyticsDetectionModelConfig
from ml.models.ultralytics.model import UltralyticsDetectionModel
from schemas.attack import AttackType
from schemas.detection import DetectionRequest
from schemas.detection_attack import DetectionAttackRequest
from schemas.explainability import DetectionExplainRequest


def _build_ultralytics_model() -> UltralyticsDetectionModel:
    return UltralyticsDetectionModel(
        UltralyticsDetectionModelConfig(
            id="yolo26n",
            name="YOLO26n · preentrenado en COCO",
            revision="ultralytics-yolo26n-coco-8.4.121",
            weights="yolo26n.pt",
            end_to_end=True,
            explainability_target_layer_indices=(-2,),
        )
    )


def _native_result(
    rows: list[list[float]],
    *,
    shape: tuple[int, int] = (320, 640),
    names=None,
):
    row_tensor = torch.tensor(rows, dtype=torch.float64).reshape(-1, 6)
    boxes = SimpleNamespace(
        xyxy=row_tensor[:, :4],
        conf=row_tensor[:, 4],
        cls=row_tensor[:, 5],
    )

    return SimpleNamespace(
        orig_shape=shape,
        boxes=boxes,
        names=names if names is not None else {0: "person", 2: "car"},
    )


def _canonical_result(image: Image.Image, confidence: float = 0.91234567):
    return DetectionResult(
        source_width=image.width,
        source_height=image.height,
        detections=(
            Detection(
                class_id=2,
                class_name="car",
                confidence=confidence,
                box=DetectionBox(1.126, 2.234, 6.875, 7.556),
            ),
        ),
    )


def test_ultralytics_adapter_converts_empty_native_result():
    result = UltralyticsDetectionAdapter().adapt(
        _native_result([]),
        source_width=640,
        source_height=320,
    )

    assert result == DetectionResult(source_width=640, source_height=320)


def test_ultralytics_adapter_converts_actual_results_object():
    native = Results(
        orig_img=np.zeros((320, 640, 3), dtype=np.uint8),
        path="source.png",
        names={0: "person", 2: "car"},
        boxes=torch.tensor(
            [[10.125, 20.25, 100.875, 200.5, 0.75, 2.0]],
            dtype=torch.float64,
        ),
    )

    result = UltralyticsDetectionAdapter().adapt(
        native,
        source_width=640,
        source_height=320,
    )

    assert result.detections == (
        Detection(
            class_id=2,
            class_name="car",
            confidence=0.75,
            box=DetectionBox(10.125, 20.25, 100.875, 200.5),
        ),
    )


def test_ultralytics_adapter_maps_multiple_detections_and_categories():
    result = UltralyticsDetectionAdapter().adapt(
        _native_result(
            [
                [10.125, 20.25, 100.875, 200.5, 0.91234567, 2.0],
                [300.2, 50.3, 500.4, 250.6, 0.33333333, 0.0],
            ]
        ),
        source_width=640,
        source_height=320,
    )

    assert len(result.detections) == 2
    assert (
        result.detections[0].class_id,
        result.detections[0].class_name,
        result.detections[0].confidence,
    ) == (2, "car", 0.91234567)
    assert result.detections[0].box.x_min == 10.125
    assert result.detections[0].box.x_max == 100.875
    assert result.detections[1].class_name == "person"


def test_ultralytics_adapter_preserves_source_coordinates_after_letterboxing():
    native = _native_result(
        [[12.1234567, 8.7654321, 630.9876543, 310.2468135, 0.7, 0.0]],
        shape=(320, 640),
    )

    result = UltralyticsDetectionAdapter().adapt(
        native,
        source_width=640,
        source_height=320,
    )

    box = result.detections[0].box
    assert (box.x_min, box.y_min, box.x_max, box.y_max) == (
        12.1234567,
        8.7654321,
        630.9876543,
        310.2468135,
    )


def test_ultralytics_adapter_clips_partially_outside_box():
    result = UltralyticsDetectionAdapter().adapt(
        _native_result([[-3.5, 4.25, 645.75, 330.0, 0.5, 2.0]]),
        source_width=640,
        source_height=320,
    )

    box = result.detections[0].box
    assert (box.x_min, box.y_min, box.x_max, box.y_max) == (
        0.0,
        4.25,
        640.0,
        320.0,
    )


@pytest.mark.parametrize(
    "row",
    [
        [20.0, 10.0, 10.0, 30.0, 0.5, 0.0],
        [10.0, 10.0, 20.0, 30.0, float("nan"), 0.0],
        [10.0, 10.0, 20.0, 30.0, 0.5, 0.25],
    ],
)
def test_ultralytics_adapter_rejects_invalid_native_detection(row):
    _prepared_receiver = UltralyticsDetectionAdapter()
    _prepared_arg_1 = _native_result([row])

    with pytest.raises(DetectionAdapterError):
        _prepared_receiver.adapt(
            _prepared_arg_1,
            source_width=640,
            source_height=320,
        )


@pytest.mark.parametrize(
    "row",
    [
        [10.0, 20.0, 10.0, 30.0, 0.5, 0.0],
        [10.0, 20.0, 20.0, 20.0, 0.5, 0.0],
        [650.0, 10.0, 660.0, 20.0, 0.5, 0.0],
        [-20.0, 10.0, -10.0, 20.0, 0.5, 0.0],
        [10.0, 330.0, 20.0, 340.0, 0.5, 0.0],
    ],
)
def test_ultralytics_adapter_skips_boxes_without_positive_clipped_area(row):
    result = UltralyticsDetectionAdapter().adapt(
        _native_result([row]),
        source_width=640,
        source_height=320,
    )

    assert result.detections == ()


def test_ultralytics_adapter_skips_degenerate_box_without_losing_valid_ones():
    rows = [[10.0, 20.0, 30.0, 40.0, 0.8, 0.0]] * 275
    rows.append([10.0, 20.0, 10.0, 40.0, 0.5, 0.0])
    rows.append([30.0, 40.0, 50.0, 60.0, 0.7, 2.0])

    result = UltralyticsDetectionAdapter().adapt(
        _native_result(rows),
        source_width=640,
        source_height=320,
    )

    assert len(result.detections) == 276
    assert result.detections[0].class_name == "person"
    assert result.detections[-1].class_name == "car"


@pytest.mark.parametrize(
    "row",
    [
        [10.0, 20.0, 10.0, 40.0, 1.1, 0.0],
        [10.0, 20.0, 10.0, 40.0, 0.5, -1.0],
        [10.0, 20.0, 10.0, 40.0, 0.5, 0.5],
        [10.0, 20.0, 10.0, 40.0, float("inf"), 0.0],
    ],
)
def test_ultralytics_adapter_does_not_hide_invalid_fields_on_degenerate_boxes(row):
    with pytest.raises(DetectionAdapterError):
        UltralyticsDetectionAdapter().adapt(
            _native_result([row]),
            source_width=640,
            source_height=320,
        )


@pytest.mark.parametrize("values", [["person"], []])
def test_ultralytics_adapter_rejects_negative_class_id_without_indexing_names(
    values,
):
    accessed_indexes = []

    class _TrackedNames(list):
        def __getitem__(self, index):
            accessed_indexes.append(index)
            return super().__getitem__(index)

    _prepared_receiver_2 = UltralyticsDetectionAdapter()
    _prepared_arg_1_2 = _native_result(
        [[10.0, 10.0, 20.0, 20.0, 0.5, -1.0]],
        names=_TrackedNames(values),
    )

    with pytest.raises(
        DetectionAdapterError,
        match=r"native detection 0 is invalid: class_id must not be negative",
    ):
        _prepared_receiver_2.adapt(
            _prepared_arg_1_2,
            source_width=640,
            source_height=320,
        )

    assert accessed_indexes == []


def test_ultralytics_adapter_rejects_mismatched_source_geometry():
    _prepared_receiver_3 = UltralyticsDetectionAdapter()
    _prepared_arg_1_3 = _native_result([[1.0, 2.0, 3.0, 4.0, 0.5, 0.0]])

    with pytest.raises(DetectionAdapterError, match="does not match"):
        _prepared_receiver_3.adapt(
            _prepared_arg_1_3,
            source_width=320,
            source_height=640,
        )


def test_ultralytics_adapter_does_not_leak_provider_values():
    result = UltralyticsDetectionAdapter().adapt(
        _native_result([[1.0, 2.0, 3.0, 4.0, 0.5, 0.0]]),
        source_width=640,
        source_height=320,
    )

    values = (
        result,
        result.detections,
        result.detections[0],
        result.detections[0].box,
        result.detections[0].confidence,
        result.detections[0].box.x_min,
    )
    assert not any(isinstance(value, (torch.Tensor, np.ndarray)) for value in values)


def test_ultralytics_adapter_converts_native_rows_once_per_result(monkeypatch):
    original_cat = torch.cat
    conversions = {"to_cpu": 0, "to_list": 0}

    class _TensorConversionSpy:
        def __init__(self, tensor):
            self.tensor = tensor

        def detach(self):
            return self

        def to(self, *, device):
            assert device == "cpu"
            conversions["to_cpu"] += 1
            return self

        def tolist(self):
            conversions["to_list"] += 1
            return self.tensor.tolist()

    def tracked_cat(tensors, *, dim):
        return _TensorConversionSpy(original_cat(tensors, dim=dim))

    monkeypatch.setattr(ultralytics_adapter.torch, "cat", tracked_cat)

    result = UltralyticsDetectionAdapter().adapt(
        _native_result(
            [
                [10.0, 20.0, 30.0, 40.0, 0.9, 0.0],
                [50.0, 60.0, 70.0, 80.0, 0.8, 2.0],
            ]
        ),
        source_width=640,
        source_height=320,
    )

    assert len(result.detections) == 2
    assert conversions == {"to_cpu": 1, "to_list": 1}


def test_ultralytics_evaluation_inference_uses_explicit_scientific_configuration():
    source_image = Image.new("RGB", (640, 320))
    native = _native_result([])
    calls = []

    class _LoadedModel:
        def __call__(self, image, **kwargs):
            calls.append((image, kwargs))
            return [native]

    model = _build_ultralytics_model()
    model.model = _LoadedModel()
    configuration = DetectionEvaluationInferenceConfiguration(
        score_floor=0.005,
    )

    result = model.detect_for_evaluation(source_image, configuration)

    assert result == DetectionResult(source_width=640, source_height=320)
    assert calls == [
        (
            source_image,
            {
                "imgsz": 640,
                "rect": True,
                "conf": 0.005,
                "max_det": 300,
                "end2end": True,
                "verbose": False,
            },
        )
    ]


def test_ultralytics_normal_detection_uses_configured_provider_geometry():
    source_image = Image.new("RGB", (640, 320))
    native = _native_result([])
    calls = []

    class _LoadedModel:
        def __call__(self, image, **kwargs):
            calls.append((image, kwargs))
            return [native]

    model = _build_ultralytics_model()
    model.model = _LoadedModel()

    model.detect(source_image)

    assert calls == [
        (
            source_image,
            {
                "imgsz": 640,
                "rect": True,
                "end2end": True,
            },
        )
    ]


@pytest.mark.asyncio
async def test_detection_service_uses_ultralytics_adapter_and_preserves_http_contract(
    monkeypatch,
):
    source_image = Image.new("RGB", (640, 320))
    native = _native_result([[10.126, 20.234, 100.875, 200.556, 0.91234567, 2.0]])
    native.plot = lambda: np.zeros((320, 640, 3), dtype=np.uint8)
    model = _build_ultralytics_model()
    inference_calls = 0

    def fake_run_inference(image):
        nonlocal inference_calls
        assert image is source_image
        inference_calls += 1
        return native

    async def fake_get_image_from_dataset(*, dataset_id, image_id, owner_id=None):
        assert owner_id is None
        assert (dataset_id, image_id) == ("dataset-1", "image-1")
        return source_image

    monkeypatch.setattr(model, "_run_inference", fake_run_inference)
    monkeypatch.setattr(
        detection_service,
        "get_image_from_dataset",
        fake_get_image_from_dataset,
    )
    monkeypatch.setattr(
        detection_service,
        "get_active_model_instance",
        lambda: model,
    )
    monkeypatch.setattr(
        detection_service,
        "create_result_directory",
        lambda prefix: ("result-1", "/tmp/result-1"),
    )
    monkeypatch.setattr(
        detection_service,
        "save_result_image",
        lambda **kwargs: "/results/result-1/annotated.png",
    )
    monkeypatch.setattr(
        detection_service,
        "save_result_metadata",
        lambda **kwargs: None,
    )

    response = await detection_service.detect_objects(
        DetectionRequest(dataset_id="dataset-1", image_id="image-1")
    )

    assert inference_calls == 1
    assert response.model_dump() == {
        "model_id": "yolo26n",
        "model_name": "YOLO26n · preentrenado en COCO",
        "detections": [
            {
                "label": "car",
                "confidence": 0.9123,
                "bbox": {
                    "x1": 10.13,
                    "y1": 20.23,
                    "x2": 100.88,
                    "y2": 200.56,
                },
            }
        ],
        "annotated_image_url": "/results/result-1/annotated.png",
    }


@pytest.mark.asyncio
async def test_detection_attack_service_keeps_canonical_detections_until_response(
    monkeypatch,
):
    source_image = Image.new("RGB", (8, 8))

    class _DetectionModel:
        id = "detector"
        name = "Detector"
        task_type = "detection"

        def supports_detection_explainability(self):
            return True

        def detect_with_visualization(self, image):
            return _canonical_result(image), image

    class _Attack:
        id = "random_noise"

        def supports_task_type(self, task_type):
            return task_type == "detection"

        def apply(self, **kwargs):
            return {
                "adversarial_image": source_image.copy(),
                "perturbation_image": source_image.copy(),
                "perturbation_linf": 0.1,
                "perturbation_l2": 0.2,
            }

    async def fake_get_image_from_dataset(**kwargs):
        return source_image

    monkeypatch.setattr(
        detection_attack_service,
        "get_image_from_dataset",
        fake_get_image_from_dataset,
    )
    monkeypatch.setattr(
        detection_attack_service,
        "get_active_model_instance",
        _DetectionModel,
    )
    monkeypatch.setattr(detection_attack_service, "get_attack", lambda value: _Attack())
    monkeypatch.setattr(
        detection_attack_service,
        "generate_detection_eigencam",
        lambda model, image: image.copy(),
    )
    monkeypatch.setattr(
        detection_attack_service,
        "create_result_directory",
        lambda prefix: ("result-1", "/tmp/result-1"),
    )
    monkeypatch.setattr(
        detection_attack_service,
        "save_result_image",
        lambda **kwargs: f"/results/result-1/{kwargs['filename']}",
    )
    monkeypatch.setattr(
        detection_attack_service,
        "save_result_metadata",
        lambda **kwargs: None,
    )

    response = await detection_attack_service.apply_detection_attack(
        DetectionAttackRequest(
            dataset_id="dataset-1",
            image_id="image-1",
            attack_type=AttackType.RANDOM_NOISE,
        )
    )

    assert response.original_detections[0].model_dump() == {
        "label": "car",
        "confidence": 0.9123,
        "bbox": {"x1": 1.13, "y1": 2.23, "x2": 6.88, "y2": 7.56},
    }
    assert response.adversarial_detections == response.original_detections


@pytest.mark.asyncio
async def test_detection_explainability_keeps_canonical_detections_until_response(
    monkeypatch,
):
    source_image = Image.new("RGB", (8, 8))

    class _DetectionModel:
        id = "detector"
        name = "Detector"
        task_type = "detection"

        def detect(self, image):
            return _canonical_result(image)

    async def fake_get_image_from_dataset(**kwargs):
        return source_image

    monkeypatch.setattr(
        explainability_service,
        "get_image_from_dataset",
        fake_get_image_from_dataset,
    )
    monkeypatch.setattr(
        explainability_service,
        "get_active_model_instance",
        _DetectionModel,
    )
    monkeypatch.setattr(
        explainability_service,
        "generate_detection_eigencam",
        lambda model, image: image.copy(),
    )
    monkeypatch.setattr(
        explainability_service,
        "create_result_directory",
        lambda prefix: ("result-1", "/tmp/result-1"),
    )
    monkeypatch.setattr(
        explainability_service,
        "save_result_image",
        lambda **kwargs: "/results/result-1/explanation.png",
    )
    monkeypatch.setattr(
        explainability_service,
        "save_result_metadata",
        lambda **kwargs: None,
    )

    response = await explainability_service.generate_detection_explainability(
        DetectionExplainRequest(dataset_id="dataset-1", image_id="image-1")
    )

    assert response.detections[0].model_dump() == {
        "label": "car",
        "confidence": 0.9123,
        "bbox": {"x1": 1.13, "y1": 2.23, "x2": 6.88, "y2": 7.56},
    }

from types import SimpleNamespace

import pytest
import torch
from ultralytics.engine.results import Results

from ml.detection_categories import (
    DetectionClass,
    DetectionClassCatalog,
    DetectionClassCatalogError,
)
from ml.models.ultralytics.adapter import UltralyticsDetectionAdapter
from ml.models.ultralytics.catalog import (
    UltralyticsClassCatalogError,
    build_ultralytics_class_catalog,
)
from ml.models.ultralytics.config import UltralyticsDetectionModelConfig
from ml.models.ultralytics.model import UltralyticsDetectionModel


def _build_ultralytics_model() -> UltralyticsDetectionModel:
    return UltralyticsDetectionModel(
        UltralyticsDetectionModelConfig(
            id="yolo26n",
            name="YOLO26n pretrained",
            revision="ultralytics-yolo26n-coco-8.4.121",
            weights="yolo26n.pt",
            explainability_target_layer_indices=(-2,),
        )
    )


def _build_catalog(names):
    return build_ultralytics_class_catalog(
        model_id="detector-1",
        model_name="Detector",
        names=names,
    )


def test_ultralytics_class_catalog_preserves_explicit_mapping_ids():
    catalog = _build_catalog({7: "truck", 2: "car"})

    assert catalog.classes == (
        DetectionClass(class_id=2, name="car"),
        DetectionClass(class_id=7, name="truck"),
    )
    assert set(catalog.class_by_id) == {2, 7}


def test_ultralytics_class_catalog_enumerates_sequence_ids():
    catalog = _build_catalog(["person", "bicycle", "car"])

    assert catalog.classes == (
        DetectionClass(class_id=0, name="person"),
        DetectionClass(class_id=1, name="bicycle"),
        DetectionClass(class_id=2, name="car"),
    )


@pytest.mark.parametrize("class_id", [-1, True])
def test_detection_class_rejects_invalid_id(class_id):
    with pytest.raises(DetectionClassCatalogError):
        DetectionClass(class_id=class_id, name="person")


def test_detection_class_catalog_rejects_duplicate_ids():
    with pytest.raises(DetectionClassCatalogError, match="IDs must be unique"):
        DetectionClassCatalog(
            model_id="detector-1",
            model_name="Detector",
            classes=(
                DetectionClass(class_id=3, name="car"),
                DetectionClass(class_id=3, name="truck"),
            ),
        )


def test_detection_class_rejects_empty_name():
    with pytest.raises(DetectionClassCatalogError, match="non-empty"):
        DetectionClass(class_id=0, name=" ")


@pytest.mark.parametrize(
    ("first_name", "second_name", "normalized_name"),
    [
        ("person", "person", "person"),
        ("Fire-Truck", "  fire__truck ", "fire truck"),
    ],
)
def test_detection_class_catalog_detects_normalized_name_collisions(
    first_name,
    second_name,
    normalized_name,
):
    catalog = _build_catalog(
        {
            2: first_name,
            9: second_name,
            12: "car",
        }
    )

    matching_collisions = [
        collision
        for collision in catalog.normalized_name_collisions
        if collision.normalized_name == normalized_name
    ]
    assert len(matching_collisions) == 1
    collision = matching_collisions[0]
    assert [candidate.class_id for candidate in collision.classes] == [2, 9]


def test_detection_class_catalog_rejects_empty_catalog():
    with pytest.raises(UltralyticsClassCatalogError, match="must not be empty"):
        _build_catalog({})


def test_ultralytics_model_gets_catalog_without_running_forward():
    forward_calls = 0

    class _LoadedYOLO:
        names = {0: "person", 2: "car"}

        def __call__(self, image):
            nonlocal forward_calls
            forward_calls += 1
            raise AssertionError("forward must not run")

    model = _build_ultralytics_model()
    model.model = _LoadedYOLO()

    catalog = model.get_detection_class_catalog()

    assert forward_calls == 0
    assert catalog.model_id == "yolo26n"
    assert catalog.classes == (
        DetectionClass(class_id=0, name="person"),
        DetectionClass(class_id=2, name="car"),
    )


class _FailingNamesYOLO:
    def __init__(self, error):
        self.error = error
        self.forward_calls = 0

    @property
    def names(self):
        raise self.error

    def __call__(self, image):
        self.forward_calls += 1
        raise AssertionError("forward must not run")


def test_ultralytics_model_wraps_missing_names_as_typed_catalog_error():
    provider_error = AttributeError("names is unavailable")
    loaded_model = _FailingNamesYOLO(provider_error)
    model = _build_ultralytics_model()
    model.model = loaded_model

    with pytest.raises(
        UltralyticsClassCatalogError,
        match=r"could not read.*yolo26n.*YOLO26n",
    ) as captured:
        model.get_detection_class_catalog()

    assert captured.value.__cause__ is provider_error
    assert loaded_model.forward_calls == 0


def test_ultralytics_model_wraps_provider_names_error_and_preserves_cause():
    provider_error = RuntimeError("provider catalog failure")
    loaded_model = _FailingNamesYOLO(provider_error)
    model = _build_ultralytics_model()
    model.model = loaded_model

    with pytest.raises(
        UltralyticsClassCatalogError,
        match=r"could not read.*yolo26n.*YOLO26n",
    ) as captured:
        model.get_detection_class_catalog()

    assert captured.value.__cause__ is provider_error
    assert loaded_model.forward_calls == 0


def test_detection_class_catalog_does_not_leak_provider_objects():
    names = {0: "person", 2: "car"}

    catalog = _build_catalog(names)
    names[0] = "mutated"

    assert catalog.classes[0].name == "person"
    assert isinstance(catalog.classes, tuple)
    assert all(
        not isinstance(value, (Results, torch.Tensor))
        for value in (catalog, *catalog.classes)
    )


def test_catalog_ids_match_prediction_adapter_class_ids():
    names = {4: "fire truck"}
    catalog = _build_catalog(names)
    native = Results(
        orig_img=torch.zeros((40, 60, 3), dtype=torch.uint8).numpy(),
        path="source.png",
        names=names,
        boxes=torch.tensor(
            [[1.0, 2.0, 30.0, 35.0, 0.75, 4.0]],
            dtype=torch.float64,
        ),
    )

    prediction = UltralyticsDetectionAdapter().adapt(
        native,
        source_width=60,
        source_height=40,
    )

    predicted = prediction.detections[0]
    catalog_class = catalog.class_by_id[predicted.class_id]
    assert (predicted.class_id, predicted.class_name) == (
        catalog_class.class_id,
        catalog_class.name,
    )


@pytest.mark.parametrize("names", ["person", 7, SimpleNamespace()])
def test_ultralytics_class_catalog_rejects_unsupported_names_format(names):
    with pytest.raises(UltralyticsClassCatalogError, match="mapping or sequence"):
        _build_catalog(names)

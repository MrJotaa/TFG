import pytest

import api.routes.datasets as datasets_route
import api.routes.models as models_route
import services.classification_label_service as label_service
from core.errors import BadRequestError
from schemas.classification_metrics import (
    ClassReference,
    LabelResolutionSummary,
)
from schemas.datasets import (
    DatasetImage,
    DatasetLabelResolutionResponse,
    DatasetResponse,
)
from schemas.models import (
    ModelClassCatalog,
    ModelClassInfo,
)
from services.classification_label_service import (
    ClassificationLabelResolver,
    normalize_classification_label,
    resolve_dataset_labels,
)


class FakeClassificationModel:
    id = "fake-classifier"
    name = "Fake Classifier"
    task_type = "classification"

    def __init__(
        self,
        labels: tuple[str, ...],
    ):
        self._labels = labels

    def get_class_labels(self) -> tuple[str, ...]:
        return self._labels


class FakeDetectionModel:
    id = "fake-detector"
    name = "Fake Detector"
    task_type = "detection"


def _create_dataset(
    images: list[DatasetImage],
) -> DatasetResponse:
    return DatasetResponse(
        id="dataset-1",
        name="Test Dataset",
        owner_id="user-1",
        image_count=len(images),
        images=images,
    )


@pytest.mark.parametrize(
    ("provided_label", "expected"),
    [
        (
            "  Motor   Scooter  ",
            "motor scooter",
        ),
        (
            "MOTOR_SCOOTER",
            "motor scooter",
        ),
        (
            "motor-scooter",
            "motor scooter",
        ),
        (
            "tabby, tabby cat",
            "tabby tabby cat",
        ),
    ],
)
def test_normalize_classification_label(
    provided_label,
    expected,
):
    assert normalize_classification_label(provided_label) == expected


def test_resolver_matches_exact_canonical_label():
    resolver = ClassificationLabelResolver(
        (
            "sports car",
            "motor scooter",
        )
    )

    match = resolver.resolve("Sports Car")

    assert match.resolved is True
    assert match.class_reference == ClassReference(
        index=0,
        label="sports car",
    )
    assert match.matched_alias == "sports car"


def test_resolver_matches_imagenet_alias():
    resolver = ClassificationLabelResolver(
        (
            "tabby, tabby cat",
            "Persian cat",
        )
    )

    match = resolver.resolve("TABBY_CAT")

    assert match.resolved is True
    assert match.class_reference == ClassReference(
        index=0,
        label="tabby, tabby cat",
    )
    assert match.matched_alias == "tabby cat"


def test_resolver_rejects_unknown_label():
    resolver = ClassificationLabelResolver(
        (
            "sports car",
            "motor scooter",
        )
    )

    match = resolver.resolve("unknown vehicle")

    assert match.resolved is False
    assert match.reason == ("La etiqueta no coincide con ninguna clase del modelo")
    assert match.candidate_classes == ()


def test_resolver_rejects_ambiguous_label():
    resolver = ClassificationLabelResolver(
        (
            "crane",
            "crane",
        )
    )

    match = resolver.resolve("crane")

    assert match.resolved is False
    assert match.reason == ("La etiqueta coincide con varias clases del modelo")

    assert match.candidate_classes == (
        ClassReference(
            index=0,
            label="crane",
        ),
        ClassReference(
            index=1,
            label="crane",
        ),
    )


def test_class_catalog_exposes_aliases():
    resolver = ClassificationLabelResolver(
        (
            "tabby, tabby cat",
            "sports car",
        )
    )

    catalog = resolver.get_catalog()

    assert catalog[0].index == 0
    assert catalog[0].label == "tabby, tabby cat"
    assert catalog[0].aliases == [
        "tabby",
        "tabby cat",
    ]

    assert catalog[1].index == 1
    assert catalog[1].label == "sports car"
    assert catalog[1].aliases == []


def test_dataset_label_resolution_reports_coverage():
    dataset = _create_dataset(
        [
            DatasetImage(
                id="image-1",
                filename="cat.png",
                label="TABBY_CAT",
            ),
            DatasetImage(
                id="image-2",
                filename="car.png",
                label="sports-car",
            ),
            DatasetImage(
                id="image-3",
                filename="unknown.png",
                label="unknown vehicle",
            ),
            DatasetImage(
                id="image-4",
                filename="unlabeled.png",
                label=None,
            ),
        ]
    )

    model = FakeClassificationModel(
        (
            "tabby, tabby cat",
            "sports car",
        )
    )

    resolution = resolve_dataset_labels(
        dataset=dataset,
        model=model,
    )

    assert resolution.summary.labeled_images == 3
    assert resolution.summary.resolved_labels == 2
    assert resolution.summary.unresolved_labels == 1
    assert resolution.summary.coverage == pytest.approx(
        2 / 3,
        abs=1e-4,
    )

    assert set(resolution.resolved_by_image_id) == {
        "image-1",
        "image-2",
    }

    unresolved = resolution.summary.unresolved[0]

    assert unresolved.image_id == "image-3"
    assert unresolved.provided_label == ("unknown vehicle")
    assert unresolved.reason == ("La etiqueta no coincide con ninguna clase del modelo")


def test_unlabeled_images_do_not_count_as_unresolved():
    dataset = _create_dataset(
        [
            DatasetImage(
                id="image-1",
                filename="unlabeled.png",
                label=None,
            ),
        ]
    )

    model = FakeClassificationModel(("sports car",))

    resolution = resolve_dataset_labels(
        dataset=dataset,
        model=model,
    )

    assert resolution.summary.labeled_images == 0
    assert resolution.summary.resolved_labels == 0
    assert resolution.summary.unresolved_labels == 0
    assert resolution.summary.coverage == 0.0


def test_active_class_catalog_rejects_detection_model(
    monkeypatch,
):
    monkeypatch.setattr(
        label_service,
        "get_active_model_instance",
        lambda: FakeDetectionModel(),
    )

    with pytest.raises(
        BadRequestError,
        match=("Endpoint '/models/active/classes' requires a classification model"),
    ):
        label_service.get_active_model_class_catalog()


def test_active_class_catalog_endpoint(
    client,
    monkeypatch,
):
    def fake_get_active_model_class_catalog():
        return ModelClassCatalog(
            model_id="resnet18",
            model_name="ResNet-18 · preentrenado en ImageNet-1K",
            total_classes=2,
            classes=[
                ModelClassInfo(
                    index=0,
                    label="tabby, tabby cat",
                    aliases=[
                        "tabby",
                        "tabby cat",
                    ],
                ),
                ModelClassInfo(
                    index=1,
                    label="sports car",
                    aliases=[],
                ),
            ],
        )

    monkeypatch.setattr(
        models_route,
        "get_active_model_class_catalog",
        fake_get_active_model_class_catalog,
    )

    response = client.get("/models/active/classes")

    assert response.status_code == 200

    data = response.json()

    assert data["model_id"] == "resnet18"
    assert data["total_classes"] == 2
    assert data["classes"][0]["index"] == 0
    assert data["classes"][0]["aliases"] == [
        "tabby",
        "tabby cat",
    ]


def test_dataset_label_resolution_endpoint(
    client,
    monkeypatch,
):
    async def fake_get_resolution(dataset_id: str):
        assert dataset_id == "dataset-1"

        return DatasetLabelResolutionResponse(
            dataset_id=dataset_id,
            model_id="resnet18",
            model_name="ResNet-18 · preentrenado en ImageNet-1K",
            summary=LabelResolutionSummary(
                labeled_images=2,
                resolved_labels=1,
                unresolved_labels=1,
                coverage=0.5,
                unresolved=[
                    {
                        "image_id": "image-2",
                        "filename": "unknown.png",
                        "provided_label": "unknown",
                        "reason": ("La etiqueta no coincide con ninguna clase del modelo"),
                    }
                ],
            ),
        )

    monkeypatch.setattr(
        datasets_route,
        "get_active_dataset_label_resolution",
        fake_get_resolution,
    )

    response = client.get("/datasets/dataset-1/label-resolution")

    assert response.status_code == 200

    data = response.json()

    assert data["dataset_id"] == "dataset-1"
    assert data["model_id"] == "resnet18"
    assert data["summary"]["labeled_images"] == 2
    assert data["summary"]["resolved_labels"] == 1
    assert data["summary"]["coverage"] == 0.5

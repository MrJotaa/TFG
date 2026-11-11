import pytest
import torch

from ml.adapters.art_classification import ARTClassificationAdapter
from ml.attacks.art_classification import BaseARTClassificationAttack
from ml.attacks.registry import (
    get_attack,
    list_attack_catalog,
    list_available_attacks,
    resolve_attack_parameters,
)
from schemas.attack import AttackCatalogItem, AttackParameters, AttackType

ART_ATTACK_TYPES = (
    AttackType.CARLINI_WAGNER,
    AttackType.DEEPFOOL,
    AttackType.HOP_SKIP_JUMP,
    AttackType.ZOO,
)

EXPECTED_TASK_COMPATIBILITY = {
    "classification": {
        AttackType.FGSM,
        AttackType.RANDOM_NOISE,
        AttackType.PGD,
        AttackType.CARLINI_WAGNER,
        AttackType.DEEPFOOL,
        AttackType.HOP_SKIP_JUMP,
        AttackType.ZOO,
    },
    "detection": {
        AttackType.RANDOM_NOISE,
        AttackType.YOLO_CONFIDENCE,
        AttackType.DETECTION_LOCALIZATION_PGD,
    },
}


class TinyClassifier(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        score = inputs.mean(dim=(1, 2, 3)) * self.scale
        return torch.stack((score, -score, score * 0.5), dim=1)


class FakeClassificationModel:
    id = "catalog-regression-classifier"
    task_type = "classification"

    def __init__(self) -> None:
        self.model = TinyClassifier()

    def is_loaded(self) -> bool:
        return True

    def get_torch_model(self) -> torch.nn.Module:
        return self.model

    def get_attack_input_shape(self) -> tuple[int, ...]:
        return (3, 4, 4)

    def get_class_labels(self) -> tuple[str, ...]:
        return ("class-a", "class-b", "class-c")

    def forward_attack_logits(self, prepared_input: torch.Tensor) -> torch.Tensor:
        return self.model(prepared_input)


def test_attack_enum_registry_and_catalog_stay_in_sync():
    registered = list_available_attacks()
    catalog = [AttackCatalogItem.model_validate(item) for item in list_attack_catalog()]

    assert len(registered) == len(set(registered))
    assert len(catalog) == len({item.id for item in catalog})
    assert set(registered) == set(AttackType)
    assert {item.id for item in catalog} == set(AttackType)


def test_legacy_attack_catalog_has_no_model_restrictions():
    for item in list_attack_catalog():
        if item["id"] == AttackType.DETECTION_LOCALIZATION_PGD:
            assert item["supported_model_ids"] == ["rf_detr_nano"]
        else:
            assert "supported_model_ids" not in item
            assert AttackCatalogItem.model_validate(item).supported_model_ids is None


def test_attack_catalog_validates_optional_model_restrictions():
    restricted = get_attack(AttackType.YOLO_CONFIDENCE).__class__()
    restricted.supported_model_ids = ("rf_detr_nano",)

    catalog_entry = restricted.catalog_entry()
    assert catalog_entry["supported_model_ids"] == ["rf_detr_nano"]
    assert (
        AttackCatalogItem.model_validate(catalog_entry).supported_model_ids
        == ["rf_detr_nano"]
    )

    for invalid in ([], [""], [" rf_detr_nano"], ["rf_detr_nano", "rf_detr_nano"]):
        with pytest.raises(ValueError):
            AttackCatalogItem.model_validate(
                {**catalog_entry, "supported_model_ids": invalid}
            )


@pytest.mark.parametrize("task_type", ["classification", "detection"])
def test_registered_task_compatibility_matrix_does_not_regress(task_type):
    compatible = {
        attack_type
        for attack_type in AttackType
        if get_attack(attack_type).supports_task_type(task_type)
    }

    assert compatible == EXPECTED_TASK_COMPATIBILITY[task_type]


def test_catalog_defaults_round_trip_through_backend_parameter_resolution():
    for attack_type in AttackType:
        attack = get_attack(attack_type)
        catalog_item = AttackCatalogItem.model_validate(attack.catalog_entry())
        defaults = {
            parameter.key: parameter.default for parameter in catalog_item.parameters
        }

        resolved = resolve_attack_parameters(
            attack,
            AttackParameters(**defaults),
        )

        assert resolved.model_dump(mode="python") == defaults


def test_catalog_parameter_metadata_is_safe_for_dynamic_ui_controls():
    for item in list_attack_catalog():
        catalog_item = AttackCatalogItem.model_validate(item)
        parameter_keys = [parameter.key for parameter in catalog_item.parameters]

        assert len(parameter_keys) == len(set(parameter_keys))

        for parameter in catalog_item.parameters:
            if parameter.control == "checkbox":
                assert parameter.value_type == "boolean"
                assert isinstance(parameter.default, bool)

            if parameter.control == "select":
                assert parameter.choices
                assert parameter.default in {
                    choice.value for choice in parameter.choices
                }

            if parameter.control in {"range", "number"}:
                assert parameter.value_type in {"float", "integer"}

            if parameter.minimum is not None:
                assert parameter.default >= parameter.minimum
            if parameter.maximum is not None:
                assert parameter.default <= parameter.maximum


@pytest.mark.parametrize("attack_type", ART_ATTACK_TYPES)
def test_art_attacks_are_versioned_and_use_common_classification_base(attack_type):
    attack = get_attack(attack_type)

    assert isinstance(attack, BaseARTClassificationAttack)
    assert attack.revision is not None
    assert attack.revision.startswith("art-1.20.1-")
    assert attack.supported_task_types == ("classification",)


@pytest.mark.parametrize("attack_type", ART_ATTACK_TYPES)
def test_art_attacks_build_with_common_adapter_without_running_generate(attack_type):
    adapter = ARTClassificationAdapter(FakeClassificationModel())
    attack = get_attack(attack_type)
    parameters = resolve_attack_parameters(attack, AttackParameters())

    art_attack = attack.build_art_attack(
        estimator=adapter.estimator,
        parameters=parameters,
    )

    assert art_attack.estimator is adapter.estimator
    assert art_attack.is_estimator_valid(
        adapter.estimator,
        art_attack.estimator_requirements,
    )


@pytest.mark.parametrize("schema", [
    "schemas.attack.AttackResponse",
    "schemas.detection_attack.DetectionAttackResponse",
    "schemas.evaluation.EvaluationResponse",
    "schemas.detection_evaluation.DetectionEvaluationResponse",
    "schemas.detection_evaluation.DetectionEvaluationHistoryItem",
    "schemas.evaluation_history.EvaluationHistoryItem",
    "schemas.individual_analysis_history.IndividualAnalysisHistoryItem",
])
def test_archived_attack_identifiers_remain_readable(schema):
    from importlib import import_module

    from pydantic import TypeAdapter

    module_name, schema_name = schema.rsplit(".", 1)
    model = getattr(import_module(module_name), schema_name)
    archived_id = "archived_attack_from_previous_version"
    annotation = model.model_fields["attack_type"].annotation
    assert TypeAdapter(annotation).validate_python(archived_id) == archived_id


@pytest.mark.parametrize("schema", [
    "schemas.attack.AttackRequest",
    "schemas.detection_attack.DetectionAttackRequest",
    "schemas.evaluation.EvaluationRequest",
    "schemas.detection_evaluation.DetectionEvaluationRequest",
    "schemas.evaluation_history.EvaluationCompatibilityRequest",
    "schemas.individual_analysis_history.IndividualAnalysisCompatibilityRequest",
])
def test_archived_attack_identifiers_cannot_be_executed(schema):
    from importlib import import_module

    from pydantic import TypeAdapter, ValidationError

    module_name, schema_name = schema.rsplit(".", 1)
    model = getattr(import_module(module_name), schema_name)
    archived_id = "archived_attack_from_previous_version"
    with pytest.raises(ValidationError):
        TypeAdapter(model.model_fields["attack_type"].annotation).validate_python(
            archived_id
        )

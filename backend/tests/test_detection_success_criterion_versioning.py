"""C6.2: el significado del éxito está versionado, sin migrar artefactos."""

import pytest
from pydantic import ValidationError

from schemas.attack import AttackParameters
from schemas.detection_attack import DetectionAttackResponse
from schemas.detection_evaluation import DetectionEvaluationResponse, DetectionEvaluationSummary
from utils.detection_metrics import (
    CURRENT_DETECTION_METRICS_SCHEMA_VERSION,
    LEGACY_DETECTION_SUCCESS,
    MATCHED_DETECTION_SUCCESS,
)


def _individual_payload():
    return {
        "result_id": "old-detection",
        "attack_type": "yolo_confidence",
        "parameters": AttackParameters().model_dump(mode="json"),
        "original_annotated_image_url": "/old/original.png",
        "adversarial_image_url": "/old/adversarial.png",
        "adversarial_annotated_image_url": "/old/annotated.png",
        "perturbation_image_url": "/old/perturbation.png",
        "original_detections": [],
        "adversarial_detections": [],
        "comparison": {
            "original_detection_count": 1,
            "adversarial_detection_count": 1,
            "detections_difference": 0,
            "average_original_confidence": 0.9,
            "average_adversarial_confidence": 0.899,
            "average_confidence_difference": 0.001,
            "attack_success": True,  # significado antiguo, NO recalcular
            "perturbation_linf": 0.03,
            "perturbation_l2": 0.1,
        },
    }


def _summary():
    return DetectionEvaluationSummary(
        total_images=1,
        total_original_detections=1,
        total_adversarial_detections=1,
        detections_difference=0,
        average_original_detections_per_image=1,
        average_adversarial_detections_per_image=1,
        average_original_confidence=0.9,
        average_adversarial_confidence=0.899,
        average_confidence_difference=0.001,
        images_with_reduced_detections=0,
        images_with_no_adversarial_detections=0,
        attack_success_rate=1.0,  # el dato legacy no se altera al leerlo
        average_perturbation_linf=0.03,
        average_perturbation_l2=0.1,
    )


def test_old_individual_payload_remains_legacy_without_recalculation():
    payload = _individual_payload()
    restored = DetectionAttackResponse.model_validate(payload)
    assert restored.metrics_schema_version == "1.0"
    assert restored.attack_success_criterion == LEGACY_DETECTION_SUCCESS
    assert restored.comparison.attack_success is True
    assert restored.comparison.average_confidence_difference == 0.001
    assert DetectionAttackResponse.model_validate_json(restored.model_dump_json()) == restored


def test_new_individual_payload_identifies_its_changed_success_definition():
    payload = {
        **_individual_payload(),
        "metrics_schema_version": CURRENT_DETECTION_METRICS_SCHEMA_VERSION,
        "attack_success_criterion": MATCHED_DETECTION_SUCCESS,
        "comparison": {**_individual_payload()["comparison"], "attack_success": False},
    }
    restored = DetectionAttackResponse.model_validate(payload)
    assert restored.metrics_schema_version == "1.1"
    assert restored.attack_success_criterion == MATCHED_DETECTION_SUCCESS
    assert restored.comparison.attack_success is False


def test_old_evaluation_payload_and_summary_are_not_reinterpreted():
    historical = DetectionEvaluationResponse(
        result_id="detection_evaluation_old",
        model_id="yolo26n",
        model_name="YOLO",
        dataset_id="old-dataset",
        dataset_revision=0,
        attack_type="yolo_confidence",
        parameters=AttackParameters(),
        summary=_summary(),
        results=[],
    )
    raw = historical.model_dump(mode="json")
    # Simulate an actual stored artifact before the new field existed.
    raw.pop("attack_success_criterion")
    restored = DetectionEvaluationResponse.model_validate(raw)
    assert restored.metrics_schema_version == "1.0"
    assert restored.attack_success_criterion == LEGACY_DETECTION_SUCCESS
    assert restored.summary.attack_success_rate == 1.0
    assert restored.model_dump(mode="json")["summary"] == raw["summary"]


def test_schema_version_and_criterion_must_agree():
    with pytest.raises(ValidationError, match="attack_success_criterion"):
        DetectionAttackResponse.model_validate({
            **_individual_payload(),
            "metrics_schema_version": "1.1",  # No criterion => legacy default
        })
    with pytest.raises(ValidationError, match="attack_success_criterion"):
        DetectionAttackResponse.model_validate({
            **_individual_payload(),
            "attack_success_criterion": MATCHED_DETECTION_SUCCESS,
        })

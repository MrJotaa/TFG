import json
from dataclasses import replace

import pytest
from pydantic import ValidationError

import ml.metrics.coco_detection as coco_detection
import services.detection_object_adversarial_analysis as object_analysis_service
from ml.metrics.coco_detection import (
    CocoEvaluationClass,
    CocoEvaluationImage,
    CocoGroundTruth,
    CocoOperatingPointConfiguration,
    CocoPrediction,
    GroundTruthMatchObservation,
    evaluate_coco_detections,
)
from ml.models.detection import Detection, DetectionBox
from schemas.detection_metrics import (
    DetectionObjectAdversarialAnalysis,
    DetectionObjectAggregate,
    DetectionObjectOutcome,
    DetectionObjectResult,
)
from services.detection_object_adversarial_analysis import (
    DetectionObjectAdversarialAnalysisInputError,
    build_detection_object_adversarial_analysis,
)


def _class(class_id=0, internal_id="category-person", name="person"):
    return CocoEvaluationClass(
        category_internal_id=internal_id,
        coco_category_id=91 + class_id,
        category_name=name,
        model_class_id=class_id,
        model_class_name=name,
    )


def _ground_truth(
    object_id,
    *,
    annotation_id=1,
    image_id="image-1",
    class_id=0,
    box=(10.0, 10.0, 30.0, 30.0),
    area=None,
    iscrowd=0,
):
    return CocoGroundTruth(
        annotation_id=annotation_id,
        object_internal_id=object_id,
        image_id=image_id,
        model_class_id=class_id,
        box=DetectionBox(*box),
        area=area if area is not None else (box[2] - box[0]) * (box[3] - box[1]),
        iscrowd=iscrowd,
    )


def _observation(
    object_id,
    *,
    matched,
    image_id="image-1",
    category=None,
    confidence=0.9,
    iou=0.8,
    box=(10.0, 10.0, 30.0, 30.0),
):
    return GroundTruthMatchObservation(
        object_internal_id=object_id,
        image_id=image_id,
        category_reference=category or _class(),
        matched=matched,
        matched_detection_confidence=confidence if matched else None,
        matched_detection_iou=iou if matched else None,
        matched_detection_box=DetectionBox(*box) if matched else None,
    )


def _analysis(
    patterns,
    *,
    include_details=True,
    ground_truths=None,
    classes=None,
    reviewed_image_ids=None,
):
    classes = tuple(classes or (_class(),))
    if ground_truths is None:
        ground_truths = tuple(
            _ground_truth(object_id, annotation_id=index)
            for index, object_id in enumerate(patterns, start=1)
        )
    original = tuple(
        _observation(
            object_id,
            matched=matched[0],
            image_id=next(
                item.image_id
                for item in ground_truths
                if item.object_internal_id == object_id
            ),
            category=next(
                item
                for item in classes
                if item.model_class_id
                == next(
                    gt.model_class_id
                    for gt in ground_truths
                    if gt.object_internal_id == object_id
                )
            ),
            confidence=0.9,
            iou=0.8,
        )
        for object_id, matched in patterns.items()
    )
    adversarial = tuple(
        _observation(
            object_id,
            matched=matched[1],
            image_id=next(
                item.image_id
                for item in ground_truths
                if item.object_internal_id == object_id
            ),
            category=next(
                item
                for item in classes
                if item.model_class_id
                == next(
                    gt.model_class_id
                    for gt in ground_truths
                    if gt.object_internal_id == object_id
                )
            ),
            confidence=0.7,
            iou=0.6,
        )
        for object_id, matched in patterns.items()
    )
    return build_detection_object_adversarial_analysis(
        ground_truths=tuple(ground_truths),
        classes=classes,
        reviewed_image_ids=(
            frozenset(reviewed_image_ids)
            if reviewed_image_ids is not None
            else frozenset(item.image_id for item in ground_truths)
        ),
        original_observations=original,
        adversarial_observations=adversarial,
        confidence_threshold=0.25,
        iou_threshold=0.5,
        include_details=include_details,
    )


@pytest.mark.parametrize(
    ("matched", "outcome"),
    [
        ((True, True), DetectionObjectOutcome.PRESERVED),
        ((True, False), DetectionObjectOutcome.SUPPRESSED),
        ((False, True), DetectionObjectOutcome.RECOVERED),
        ((False, False), DetectionObjectOutcome.MISSED_BOTH),
    ],
)
def test_object_transition_classifies_each_outcome(matched, outcome):
    analysis = _analysis({"persistent-object": matched})

    assert analysis.objects[0].outcome is outcome


def test_all_four_outcomes_coexist_and_aggregate_globally():
    analysis = _analysis(
        {
            "preserved": (True, True),
            "suppressed": (True, False),
            "recovered": (False, True),
            "missed": (False, False),
        }
    )

    assert analysis.summary.total_objects == 4
    assert analysis.summary.preserved_count == 1
    assert analysis.summary.suppressed_count == 1
    assert analysis.summary.recovered_count == 1
    assert analysis.summary.missed_both_count == 1
    assert analysis.summary.originally_detected_count == 2
    assert analysis.summary.adversarially_detected_count == 2
    assert analysis.summary.preservation_rate == 0.5
    assert analysis.summary.suppression_rate == 0.5


def test_preserved_object_exposes_full_precision_drops_rounded_publicly():
    category = _class()
    ground_truth = _ground_truth("object")
    analysis = build_detection_object_adversarial_analysis(
        ground_truths=(ground_truth,),
        classes=(category,),
        reviewed_image_ids=frozenset({"image-1"}),
        original_observations=(
            _observation(
                "object",
                matched=True,
                category=category,
                confidence=0.912345678,
                iou=0.876543219,
            ),
        ),
        adversarial_observations=(
            _observation(
                "object",
                matched=True,
                category=category,
                confidence=0.712345611,
                iou=0.676543111,
            ),
        ),
        confidence_threshold=0.25,
        iou_threshold=0.5,
        include_details=True,
    )

    assert analysis.objects[0].confidence_drop == 0.2
    assert analysis.objects[0].iou_drop == 0.2
    assert analysis.summary.mean_confidence_drop_preserved == 0.2
    assert analysis.summary.mean_iou_drop_preserved == 0.2


def test_preserved_drops_are_consistent_with_public_precision_observations():
    category = _class()
    analysis = build_detection_object_adversarial_analysis(
        ground_truths=(_ground_truth("object"),),
        classes=(category,),
        reviewed_image_ids=frozenset({"image-1"}),
        original_observations=(
            _observation(
                "object",
                matched=True,
                category=category,
                confidence=0.2577291507357619,
                iou=0.5288604654318989,
            ),
        ),
        adversarial_observations=(
            _observation(
                "object",
                matched=True,
                category=category,
                confidence=0.25064859980091997,
                iou=0.5258257547804502,
            ),
        ),
        confidence_threshold=0.25,
        iou_threshold=0.5,
        include_details=True,
    )

    result = analysis.objects[0]
    assert result.confidence_drop == round(
        result.original_observation.confidence
        - result.adversarial_observation.confidence,
        6,
    )
    assert result.iou_drop == round(
        result.original_observation.iou_with_ground_truth
        - result.adversarial_observation.iou_with_ground_truth,
        6,
    )
    assert analysis.summary.mean_confidence_drop_preserved == result.confidence_drop
    assert analysis.summary.mean_iou_drop_preserved == result.iou_drop


def test_negative_preserved_drops_are_not_clamped():
    category = _class()
    analysis = build_detection_object_adversarial_analysis(
        ground_truths=(_ground_truth("object"),),
        classes=(category,),
        reviewed_image_ids=frozenset({"image-1"}),
        original_observations=(
            _observation(
                "object", matched=True, category=category, confidence=0.5, iou=0.6
            ),
        ),
        adversarial_observations=(
            _observation(
                "object", matched=True, category=category, confidence=0.7, iou=0.8
            ),
        ),
        confidence_threshold=0.25,
        iou_threshold=0.5,
        include_details=True,
    )

    assert analysis.objects[0].confidence_drop == pytest.approx(-0.2)
    assert analysis.objects[0].iou_drop == pytest.approx(-0.2)


@pytest.mark.parametrize(
    "matched",
    [(True, False), (False, True), (False, False)],
)
def test_non_preserved_outcomes_have_null_deltas(matched):
    result = _analysis({"object": matched}).objects[0]

    assert result.confidence_drop is None
    assert result.iou_drop is None


def test_class_aggregation_includes_resolved_zero_support_class():
    classes = (_class(), _class(7, "category-empty", "empty"))
    analysis = _analysis({"object": (True, False)}, classes=classes)

    assert [item.category.model_class_id for item in analysis.by_class] == [0, 7]
    assert analysis.by_class[0].suppressed_count == 1
    assert analysis.by_class[1].total_objects == 0
    assert analysis.by_class[1].preservation_rate is None


def test_size_aggregation_uses_exact_coco_boundaries():
    ground_truths = (
        _ground_truth("small", annotation_id=1, area=32**2 - 0.001),
        _ground_truth("medium-low", annotation_id=2, area=32**2),
        _ground_truth("medium-high", annotation_id=3, area=96**2 - 0.001),
        _ground_truth("large", annotation_id=4, area=96**2),
    )
    analysis = _analysis(
        {item.object_internal_id: (True, True) for item in ground_truths},
        ground_truths=ground_truths,
    )

    assert [item.total_objects for item in analysis.by_size] == [1, 2, 1]


def test_image_aggregation_includes_reviewed_empty_images():
    ground_truths = (
        _ground_truth("suppressed", image_id="image-b"),
        _ground_truth("preserved", annotation_id=2, image_id="image-c"),
    )
    analysis = _analysis(
        {"suppressed": (True, False), "preserved": (True, True)},
        ground_truths=ground_truths,
        reviewed_image_ids={"image-a", "image-b", "image-c"},
    )

    assert [item.image_id for item in analysis.by_image] == [
        "image-a",
        "image-b",
        "image-c",
    ]
    assert analysis.by_image[0].total_objects == 0
    assert analysis.by_image[0].object_level_attack_success is None
    assert analysis.by_image[1].object_level_attack_success is True
    assert analysis.by_image[2].object_level_attack_success is False


def test_crowd_ground_truth_is_excluded_from_object_universe():
    category = _class()
    analysis = build_detection_object_adversarial_analysis(
        ground_truths=(_ground_truth("crowd", iscrowd=1),),
        classes=(category,),
        reviewed_image_ids=frozenset({"image-1"}),
        original_observations=(),
        adversarial_observations=(),
        confidence_threshold=0.25,
        iou_threshold=0.5,
        include_details=True,
    )

    assert analysis.total_evaluable_objects == 0
    assert analysis.objects == ()


def test_details_flag_omits_only_public_per_object_and_image_materialization():
    detailed = _analysis({"object": (True, False)}, include_details=True)
    compact = _analysis({"object": (True, False)}, include_details=False)

    assert compact.details_included is False
    assert compact.objects is None
    assert compact.by_image is None
    assert compact.summary == detailed.summary
    assert compact.by_class == detailed.by_class
    assert compact.by_size == detailed.by_size


@pytest.mark.parametrize("include_details", [False, True])
@pytest.mark.parametrize(
    ("confidence", "iou", "field_name"),
    [
        (0.1, 0.8, "confidence"),
        (0.9, 0.4, "IoU"),
    ],
)
def test_match_below_operating_point_is_rejected_in_both_detail_modes(
    include_details,
    confidence,
    iou,
    field_name,
):
    category = _class()
    matched = _observation(
        "object",
        matched=True,
        category=category,
        confidence=confidence,
        iou=iou,
    )

    with pytest.raises(
        DetectionObjectAdversarialAnalysisInputError,
        match=f"{field_name} is below the operating point",
    ):
        build_detection_object_adversarial_analysis(
            ground_truths=(_ground_truth("object"),),
            classes=(category,),
            reviewed_image_ids=frozenset({"image-1"}),
            original_observations=(matched,),
            adversarial_observations=(matched,),
            confidence_threshold=0.25,
            iou_threshold=0.5,
            include_details=include_details,
        )


def test_compact_mode_does_not_construct_public_object_schemas(monkeypatch):
    def forbidden_object_schema(**kwargs):
        raise AssertionError("compact mode must not materialize object schemas")

    monkeypatch.setattr(
        object_analysis_service,
        "DetectionObjectResult",
        forbidden_object_schema,
    )

    analysis = _analysis({"object": (True, True)}, include_details=False)

    assert analysis.objects is None


def test_input_order_does_not_change_canonical_json():
    ground_truths = (
        _ground_truth("z", image_id="image-b", annotation_id=2),
        _ground_truth("a", image_id="image-a", annotation_id=1),
    )
    patterns = {"z": (True, False), "a": (False, True)}
    first = _analysis(patterns, ground_truths=ground_truths)
    second = _analysis(
        dict(reversed(tuple(patterns.items()))),
        ground_truths=tuple(reversed(ground_truths)),
    )

    assert first.model_dump_json() == second.model_dump_json()
    assert [(item.image_id, item.object_id) for item in first.objects] == [
        ("image-a", "a"),
        ("image-b", "z"),
    ]


@pytest.mark.parametrize("condition", ["missing", "additional", "duplicate"])
def test_comparison_rejects_non_exact_observation_sets(condition):
    category = _class()
    ground_truth = _ground_truth("object")
    valid = _observation("object", matched=False, category=category)
    original = (valid,)
    if condition == "missing":
        original = ()
    elif condition == "additional":
        original = (valid, _observation("extra", matched=False, category=category))
    else:
        original = (valid, valid)

    with pytest.raises(DetectionObjectAdversarialAnalysisInputError):
        build_detection_object_adversarial_analysis(
            ground_truths=(ground_truth,),
            classes=(category,),
            reviewed_image_ids=frozenset({"image-1"}),
            original_observations=original,
            adversarial_observations=(valid,),
            confidence_threshold=0.25,
            iou_threshold=0.5,
            include_details=False,
        )


def test_comparison_rejects_incoherent_image_or_category_reference():
    category = _class()
    wrong = _observation("object", matched=False, image_id="wrong", category=category)

    with pytest.raises(
        DetectionObjectAdversarialAnalysisInputError, match="references"
    ):
        build_detection_object_adversarial_analysis(
            ground_truths=(_ground_truth("object"),),
            classes=(category,),
            reviewed_image_ids=frozenset({"image-1"}),
            original_observations=(wrong,),
            adversarial_observations=(wrong,),
            confidence_threshold=0.25,
            iou_threshold=0.5,
            include_details=False,
        )


def test_comparison_rejects_match_field_invariants():
    category = _class()
    incomplete = GroundTruthMatchObservation(
        object_internal_id="object",
        image_id="image-1",
        category_reference=category,
        matched=True,
        matched_detection_confidence=None,
        matched_detection_iou=None,
        matched_detection_box=None,
    )

    with pytest.raises(
        DetectionObjectAdversarialAnalysisInputError, match="incomplete"
    ):
        build_detection_object_adversarial_analysis(
            ground_truths=(_ground_truth("object"),),
            classes=(category,),
            reviewed_image_ids=frozenset({"image-1"}),
            original_observations=(incomplete,),
            adversarial_observations=(incomplete,),
            confidence_threshold=0.25,
            iou_threshold=0.5,
            include_details=False,
        )


@pytest.mark.parametrize("invalid_field", ["confidence", "iou", "area"])
def test_comparison_wraps_wrong_numeric_types_in_typed_input_error(invalid_field):
    category = _class()
    ground_truth = _ground_truth("object")
    original = _observation("object", matched=True, category=category)
    if invalid_field == "confidence":
        original = replace(original, matched_detection_confidence="bad")
    elif invalid_field == "iou":
        original = replace(original, matched_detection_iou="bad")
    else:
        ground_truth = replace(ground_truth, area="bad")

    with pytest.raises(
        DetectionObjectAdversarialAnalysisInputError,
        match="must be a real number",
    ):
        build_detection_object_adversarial_analysis(
            ground_truths=(ground_truth,),
            classes=(category,),
            reviewed_image_ids=frozenset({"image-1"}),
            original_observations=(original,),
            adversarial_observations=(
                _observation("object", matched=True, category=category),
            ),
            confidence_threshold=0.25,
            iou_threshold=0.5,
            include_details=False,
        )


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("confidence_threshold", "bad"),
        ("iou_threshold", True),
        ("include_details", 1),
    ],
)
def test_comparison_rejects_wrong_control_types_with_typed_error(argument, value):
    kwargs = {
        "ground_truths": (_ground_truth("object"),),
        "classes": (_class(),),
        "reviewed_image_ids": frozenset({"image-1"}),
        "original_observations": (_observation("object", matched=False),),
        "adversarial_observations": (_observation("object", matched=False),),
        "confidence_threshold": 0.25,
        "iou_threshold": 0.5,
        "include_details": False,
    }
    kwargs[argument] = value

    with pytest.raises(DetectionObjectAdversarialAnalysisInputError):
        build_detection_object_adversarial_analysis(**kwargs)


def test_comparison_rejects_ground_truth_without_persisted_id_and_never_falls_back():
    ground_truth = replace(
        _ground_truth("persisted-id", annotation_id=77), object_internal_id=None
    )

    with pytest.raises(
        DetectionObjectAdversarialAnalysisInputError,
        match="object_internal_id must be a non-empty string",
    ):
        build_detection_object_adversarial_analysis(
            ground_truths=(ground_truth,),
            classes=(_class(),),
            reviewed_image_ids=frozenset({"image-1"}),
            original_observations=(),
            adversarial_observations=(),
            confidence_threshold=0.25,
            iou_threshold=0.5,
            include_details=False,
        )


def test_schema_rejects_impossible_object_outcome_combination():
    valid = _analysis({"object": (True, False)}).objects[0].model_dump()
    valid["outcome"] = "preserved"

    with pytest.raises(ValidationError):
        DetectionObjectResult.model_validate(valid)


def test_schema_rejects_size_bucket_inconsistent_with_authoritative_area():
    valid = _analysis({"object": (True, True)}).objects[0].model_dump()
    valid["size_bucket"] = "large"

    with pytest.raises(ValidationError, match="size_bucket"):
        DetectionObjectResult.model_validate(valid)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [("total_objects", 2), ("originally_detected_count", 0), ("suppression_rate", 0.0)],
)
def test_aggregate_schema_rejects_incoherent_counts_and_rates(field_name, value):
    data = _analysis({"object": (True, False)}).summary.model_dump()
    data[field_name] = value

    with pytest.raises(ValidationError):
        DetectionObjectAggregate.model_validate(data)


def test_analysis_schema_rejects_dimension_that_does_not_sum_globally():
    data = _analysis({"object": (True, False)}).model_dump()
    data["by_size"][0]["suppressed_count"] = 0

    with pytest.raises(ValidationError):
        DetectionObjectAdversarialAnalysis.model_validate(data)


def test_analysis_schema_rejects_match_below_declared_operating_point():
    data = _analysis({"object": (True, True)}).model_dump()
    data["objects"][0]["original_observation"]["confidence"] = 0.1
    data["objects"][0]["confidence_drop"] = -0.6

    with pytest.raises(ValidationError, match="below the operating point"):
        DetectionObjectAdversarialAnalysis.model_validate(data)


def test_zero_denominators_are_null_not_zero():
    analysis = _analysis({"object": (False, False)})

    assert analysis.summary.preservation_rate is None
    assert analysis.summary.suppression_rate is None
    assert analysis.summary.original_detection_rate == 0.0
    assert analysis.summary.mean_confidence_drop_preserved is None


def _evaluate_adapter(*, prediction_confidence=0.9, prediction_box=(10, 10, 30, 30)):
    category = _class(7, "category-persisted", "persisted")
    ground_truth = _ground_truth(
        "mongo-object-id",
        class_id=7,
        annotation_id=991,
    )
    prediction = CocoPrediction(
        image_id="image-1",
        detection=Detection(
            class_id=7,
            class_name="persisted",
            confidence=prediction_confidence,
            box=DetectionBox(*prediction_box),
        ),
    )
    return evaluate_coco_detections(
        images=(CocoEvaluationImage("image-1", 100, 100),),
        classes=(category,),
        ground_truths=(ground_truth,),
        predictions=(prediction,),
        operating_point=CocoOperatingPointConfiguration(0.25, 0.5),
    )


def test_adapter_exposes_persisted_id_and_official_iou_without_ephemeral_ids():
    analysis = _evaluate_adapter(prediction_box=(12, 10, 32, 30))
    observation = analysis.ground_truth_observations[0]

    assert observation.object_internal_id == "mongo-object-id"
    assert observation.category_reference.model_class_id == 7
    assert observation.matched_detection_iou == pytest.approx(360 / 440)
    assert not hasattr(observation, "coco_ground_truth_id")


def test_adapter_uses_same_inclusive_confidence_boundary_as_operating_point():
    analysis = _evaluate_adapter(prediction_confidence=0.25)

    assert analysis.operating_point.true_positives == 1
    assert analysis.ground_truth_observations[0].matched is True


def test_adapter_turns_official_low_score_match_into_unmatched_observation():
    analysis = _evaluate_adapter(prediction_confidence=0.249999)

    assert analysis.operating_point.false_negatives == 1
    assert analysis.ground_truth_observations[0].matched is False
    assert analysis.ground_truth_observations[0].matched_detection_iou is None


def test_adapter_uses_official_iou_threshold_for_object_match():
    analysis = _evaluate_adapter(prediction_box=(17, 10, 37, 30))

    assert analysis.operating_point.false_negatives == 1
    assert analysis.ground_truth_observations[0].matched is False


def test_adapter_extracts_observations_during_single_cocoeval(monkeypatch):
    calls = 0
    real_evaluate = coco_detection.COCOeval.evaluate

    def tracked_evaluate(self):
        nonlocal calls
        calls += 1
        return real_evaluate(self)

    monkeypatch.setattr(coco_detection.COCOeval, "evaluate", tracked_evaluate)

    analysis = _evaluate_adapter()

    assert calls == 1
    assert len(analysis.ground_truth_observations) == 1


def test_match_observation_is_an_immutable_project_dataclass():
    observation = _evaluate_adapter().ground_truth_observations[0]

    with pytest.raises((AttributeError, TypeError)):
        observation.matched = False


@pytest.mark.parametrize("include_details", [False, True])
def test_complete_analysis_serializes_to_strict_json_without_provider_objects(
    include_details,
):
    analysis = _analysis(
        {"preserved": (True, True), "missed": (False, False)},
        include_details=include_details,
    )

    payload = analysis.model_dump_json()
    decoded = json.loads(payload, parse_constant=lambda value: pytest.fail(value))
    assert decoded["methodology_version"] == ("coco_category_aware_gt_transitions_v1")
    assert "COCOeval" not in payload

    def numeric_values(value):
        if isinstance(value, dict):
            for nested in value.values():
                yield from numeric_values(nested)
        elif isinstance(value, list):
            for nested in value:
                yield from numeric_values(nested)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            yield value

    assert -1 not in tuple(numeric_values(decoded))

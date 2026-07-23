from dataclasses import FrozenInstanceError

import pytest

from ml.detection_categories import (
    DetectionCategoryMatch,
    DetectionCategoryResolution,
    DetectionCategoryResolutionError,
    DetectionCategoryResolutionReason,
    DetectionCategoryResolutionStatus,
    DetectionClass,
    DetectionClassCatalog,
    normalize_detection_category_name,
    resolve_detection_categories,
)
from ml.models.detection import Detection, DetectionBox, DetectionResult
from services.annotation_importers.base import CanonicalCategory


def _catalog(*classes: tuple[int, str]) -> DetectionClassCatalog:
    return DetectionClassCatalog(
        model_id="detector-1",
        model_name="Detector",
        classes=tuple(
            DetectionClass(class_id=class_id, name=name) for class_id, name in classes
        ),
    )


def _category(
    internal_id: str,
    external_id: int,
    name: str,
) -> CanonicalCategory:
    return CanonicalCategory(
        internal_id=internal_id,
        external_id=external_id,
        name=name,
        supercategory=None,
    )


def test_resolution_matches_name_without_comparing_coco_and_model_ids():
    catalog = _catalog((0, "person"), (1, "bicycle"))
    category = _category("category-person", 1, "person")

    resolution = resolve_detection_categories(
        catalog=catalog,
        categories=(category,),
    )

    match = resolution.matches[0]
    assert category.external_id == 1
    assert match.resolved_class == DetectionClass(class_id=0, name="person")
    assert match.resolved_class.class_id != category.external_id
    assert resolution.class_id_by_category_internal_id == {"category-person": 0}


def test_resolution_matches_arbitrary_external_id_by_name():
    resolution = resolve_detection_categories(
        catalog=_catalog((7, "truck")),
        categories=(_category("category-truck", 9876, "truck"),),
    )

    assert resolution.matches[0].resolved_class.class_id == 7


@pytest.mark.parametrize(
    ("dataset_name", "model_name", "normalized"),
    [
        ("ＰＥＲＳＯＮ", "person", "person"),
        ("  Fire___TRUCK  ", "fire-truck", "fire truck"),
        ("traffic   light", "Traffic Light", "traffic light"),
    ],
)
def test_resolution_normalizes_unicode_case_and_syntactic_spacing(
    dataset_name,
    model_name,
    normalized,
):
    resolution = resolve_detection_categories(
        catalog=_catalog((3, model_name)),
        categories=(_category("category-1", 42, dataset_name),),
    )

    match = resolution.matches[0]
    assert normalize_detection_category_name(dataset_name) == normalized
    assert match.normalized_category_name == normalized
    assert match.resolved is True


def test_resolution_reports_unknown_category_without_exception():
    resolution = resolve_detection_categories(
        catalog=_catalog((0, "person")),
        categories=(_category("category-unknown", 50, "spaceship"),),
    )

    match = resolution.matches[0]
    assert match.status is DetectionCategoryResolutionStatus.UNRESOLVED
    assert match.reason is DetectionCategoryResolutionReason.NO_MATCHING_MODEL_CLASS
    assert match.candidate_classes == ()


def test_resolution_does_not_use_fuzzy_matching():
    resolution = resolve_detection_categories(
        catalog=_catalog((0, "person")),
        categories=(_category("category-typo", 1, "persn"),),
    )

    assert resolution.resolved_count == 0
    assert resolution.matches[0].reason is (
        DetectionCategoryResolutionReason.NO_MATCHING_MODEL_CLASS
    )


def test_resolution_reports_normalized_model_name_ambiguity():
    resolution = resolve_detection_categories(
        catalog=_catalog((2, "fire-truck"), (9, "Fire__Truck")),
        categories=(_category("category-fire-truck", 90, "fire truck"),),
    )

    match = resolution.matches[0]
    assert match.status is DetectionCategoryResolutionStatus.AMBIGUOUS
    assert match.reason is (
        DetectionCategoryResolutionReason.AMBIGUOUS_MODEL_CLASS_NAME
    )
    assert [candidate.class_id for candidate in match.candidate_classes] == [2, 9]
    assert resolution.ambiguity_count == 1


def test_resolution_reuses_one_candidate_tuple_for_equivalent_ambiguities():
    catalog = _catalog(
        (11, "fire-truck"),
        (2, "Fire__Truck"),
        (7, "FIRE TRUCK"),
    )
    categories = tuple(
        _category(f"category-{index}", index, "fire truck") for index in range(6)
    )

    resolution = resolve_detection_categories(
        catalog=catalog,
        categories=categories,
    )

    candidate_tuples = [
        match.candidate_classes
        for match in resolution.matches
        if match.status is DetectionCategoryResolutionStatus.AMBIGUOUS
    ]
    assert len(candidate_tuples) == len(categories)
    assert all(candidates is candidate_tuples[0] for candidates in candidate_tuples)
    assert tuple(candidate.class_id for candidate in candidate_tuples[0]) == (2, 7, 11)
    assert len({id(candidates) for candidates in candidate_tuples}) == 1
    assert len({id(candidate) for candidate in candidate_tuples[0]}) == 3


def test_resolution_marks_all_categories_competing_for_one_class_as_conflicts():
    resolution = resolve_detection_categories(
        catalog=_catalog((0, "person")),
        categories=(
            _category("category-a", 1, "person"),
            _category("category-b", 200, " PERSON "),
        ),
    )

    assert resolution.resolved_count == 0
    assert resolution.conflict_count == 2
    assert all(
        match.status is DetectionCategoryResolutionStatus.CONFLICT
        for match in resolution.matches
    )
    assert all(
        match.reason
        is DetectionCategoryResolutionReason.MODEL_CLASS_CLAIMED_BY_MULTIPLE_CATEGORIES
        for match in resolution.matches
    )
    assert resolution.class_id_by_category_internal_id == {}


def _resolved_match(
    category_internal_id: str,
    resolved_class: DetectionClass,
    *,
    category_name: str = "person",
) -> DetectionCategoryMatch:
    return DetectionCategoryMatch(
        category_internal_id=category_internal_id,
        coco_category_id=1,
        category_name=category_name,
        normalized_category_name=normalize_detection_category_name(category_name),
        status=DetectionCategoryResolutionStatus.RESOLVED,
        resolved_class=resolved_class,
    )


def test_resolution_contract_rejects_duplicate_resolved_class_ids():
    model_class = DetectionClass(class_id=0, name="person")
    catalog = _catalog((0, "person"))

    with pytest.raises(
        DetectionCategoryResolutionError,
        match="resolved class ID 0 is claimed by multiple",
    ):
        DetectionCategoryResolution(
            catalog=catalog,
            matches=(
                _resolved_match("category-a", model_class),
                _resolved_match("category-b", model_class),
            ),
        )


def test_resolution_contract_rejects_resolved_class_outside_catalog():
    with pytest.raises(
        DetectionCategoryResolutionError,
        match="resolved class ID 9.*not present in catalog",
    ):
        DetectionCategoryResolution(
            catalog=_catalog((0, "person")),
            matches=(
                _resolved_match(
                    "category-person",
                    DetectionClass(class_id=9, name="person"),
                ),
            ),
        )


def test_resolution_contract_rejects_resolved_class_name_mismatch():
    with pytest.raises(
        DetectionCategoryResolutionError,
        match="resolved class ID 0 name does not match",
    ):
        DetectionCategoryResolution(
            catalog=_catalog((0, "car")),
            matches=(
                _resolved_match(
                    "category-person",
                    DetectionClass(class_id=0, name="car"),
                ),
            ),
        )


def _ambiguous_match(
    *candidates: DetectionClass,
) -> DetectionCategoryMatch:
    return DetectionCategoryMatch(
        category_internal_id="category-person",
        coco_category_id=1,
        category_name="person",
        normalized_category_name="person",
        status=DetectionCategoryResolutionStatus.AMBIGUOUS,
        reason=DetectionCategoryResolutionReason.AMBIGUOUS_MODEL_CLASS_NAME,
        candidate_classes=candidates,
    )


def test_resolution_contract_rejects_candidate_outside_catalog():
    with pytest.raises(
        DetectionCategoryResolutionError,
        match="candidate class ID 9.*not present in catalog",
    ):
        DetectionCategoryResolution(
            catalog=_catalog((0, "person")),
            matches=(
                _ambiguous_match(
                    DetectionClass(class_id=0, name="person"),
                    DetectionClass(class_id=9, name="person"),
                ),
            ),
        )


def test_resolution_contract_rejects_candidate_name_mismatch():
    with pytest.raises(
        DetectionCategoryResolutionError,
        match="candidate class ID 1 name does not match",
    ):
        DetectionCategoryResolution(
            catalog=_catalog((0, "person"), (1, "car")),
            matches=(
                _ambiguous_match(
                    DetectionClass(class_id=0, name="person"),
                    DetectionClass(class_id=1, name="car"),
                ),
            ),
        )


def test_category_match_rejects_duplicate_candidate_ids():
    duplicated = DetectionClass(class_id=0, name="person")

    with pytest.raises(
        DetectionCategoryResolutionError,
        match="candidate_classes must contain unique class IDs",
    ):
        _ambiguous_match(duplicated, duplicated)


def test_resolution_contract_accepts_valid_resolver_output():
    resolution = resolve_detection_categories(
        catalog=_catalog((0, "person"), (4, "car")),
        categories=(
            _category("category-person", 1, "person"),
            _category("category-car", 3, "car"),
        ),
    )

    assert resolution.class_id_by_category_internal_id == {
        "category-car": 4,
        "category-person": 0,
    }


def test_resolution_reports_partial_coverage_and_builds_internal_id_mapping():
    resolution = resolve_detection_categories(
        catalog=_catalog((0, "person"), (4, "car"), (8, "dog")),
        categories=(
            _category("category-person", 1, "person"),
            _category("category-car", 3, "CAR"),
            _category("category-unknown", 99, "spaceship"),
        ),
    )

    assert resolution.total_categories == 3
    assert resolution.resolved_count == 2
    assert resolution.unresolved_count == 1
    assert resolution.ambiguity_count == 0
    assert resolution.conflict_count == 0
    assert resolution.coverage == pytest.approx(2 / 3)
    assert resolution.class_id_by_category_internal_id == {
        "category-car": 4,
        "category-person": 0,
    }


def test_resolution_accepts_dataset_without_categories():
    resolution = resolve_detection_categories(
        catalog=_catalog((0, "person")),
        categories=(),
    )

    assert resolution.matches == ()
    assert resolution.total_categories == 0
    assert resolution.resolved_count == 0
    assert resolution.unresolved_count == 0
    assert resolution.coverage == 0.0
    assert resolution.class_id_by_category_internal_id == {}


def test_resolution_order_is_deterministic():
    catalog = _catalog((0, "person"), (4, "car"), (8, "dog"))
    categories = (
        _category("category-dog", 8, "dog"),
        _category("category-person", 1, "person"),
        _category("category-car", 3, "car"),
    )

    first = resolve_detection_categories(
        catalog=catalog,
        categories=categories,
    )
    second = resolve_detection_categories(
        catalog=catalog,
        categories=tuple(reversed(categories)),
    )

    assert first.matches == second.matches
    assert [match.category_internal_id for match in first.matches] == [
        "category-car",
        "category-dog",
        "category-person",
    ]


def test_resolution_result_is_immutable_and_does_not_share_input_collection():
    source_categories = [_category("category-person", 1, "person")]
    resolution = resolve_detection_categories(
        catalog=_catalog((0, "person")),
        categories=source_categories,
    )
    source_categories.clear()

    assert len(resolution.matches) == 1
    assert isinstance(resolution.matches, tuple)
    with pytest.raises(FrozenInstanceError):
        resolution.coverage = 0.0


def test_resolution_does_not_modify_prediction_contract_precision():
    prediction = DetectionResult(
        source_width=100,
        source_height=80,
        detections=(
            Detection(
                class_id=0,
                class_name="person",
                confidence=0.91234567,
                box=DetectionBox(1.123456, 2.234567, 30.345678, 40.456789),
            ),
        ),
    )

    resolution = resolve_detection_categories(
        catalog=_catalog((0, "person")),
        categories=(_category("category-person", 42, "person"),),
    )

    assert resolution.class_id_by_category_internal_id["category-person"] == 0
    assert prediction.detections[0].confidence == 0.91234567
    assert prediction.detections[0].box == DetectionBox(
        1.123456,
        2.234567,
        30.345678,
        40.456789,
    )

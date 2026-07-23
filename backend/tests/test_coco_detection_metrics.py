import math

import pytest

from ml.metrics.coco_detection import (
    CocoEvaluationClass,
    CocoEvaluationImage,
    CocoGroundTruth,
    CocoOperatingPointConfiguration,
    CocoPrediction,
    _build_coco_apis,
    evaluate_coco_detections,
)
from ml.models.detection import Detection, DetectionBox


def _image(image_id="image-1", width=200, height=200):
    return CocoEvaluationImage(
        image_id=image_id,
        width=width,
        height=height,
    )


def _class(
    model_class_id=0,
    *,
    internal_id="category-person",
    coco_category_id=91,
    name="person",
):
    return CocoEvaluationClass(
        category_internal_id=internal_id,
        coco_category_id=coco_category_id,
        category_name=name,
        model_class_id=model_class_id,
        model_class_name=name,
    )


def _ground_truth(
    box,
    *,
    annotation_id=1,
    image_id="image-1",
    model_class_id=0,
    area=None,
    iscrowd=0,
):
    width = box[2] - box[0]
    height = box[3] - box[1]
    return CocoGroundTruth(
        annotation_id=annotation_id,
        image_id=image_id,
        model_class_id=model_class_id,
        box=DetectionBox(*box),
        area=width * height if area is None else area,
        iscrowd=iscrowd,
    )


def _prediction(
    box,
    *,
    image_id="image-1",
    model_class_id=0,
    confidence=0.9,
):
    return CocoPrediction(
        image_id=image_id,
        detection=Detection(
            class_id=model_class_id,
            class_name="person",
            confidence=confidence,
            box=DetectionBox(*box),
        ),
    )


def _evaluate(
    *,
    images=None,
    classes=None,
    ground_truths=None,
    predictions=None,
    confidence_threshold=0.25,
    iou_threshold=0.5,
):
    return evaluate_coco_detections(
        images=tuple(images or (_image(),)),
        classes=tuple(classes or (_class(),)),
        ground_truths=tuple(
            ground_truths
            if ground_truths is not None
            else (_ground_truth((10, 10, 30, 30)),)
        ),
        predictions=tuple(
            predictions if predictions is not None else (_prediction((10, 10, 30, 30)),)
        ),
        operating_point=CocoOperatingPointConfiguration(
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
        ),
    )


def test_coco_metrics_perfect_detection():
    analysis = _evaluate()

    assert analysis.aggregate.map_50_95 == pytest.approx(1.0)
    assert analysis.aggregate.ap_50 == pytest.approx(1.0)
    assert analysis.aggregate.ap_75 == pytest.approx(1.0)
    assert analysis.aggregate.ar_1 == pytest.approx(1.0)
    assert analysis.aggregate.ar_10 == pytest.approx(1.0)
    assert analysis.aggregate.ar_100 == pytest.approx(1.0)
    assert analysis.aggregate.ap_small == pytest.approx(1.0)
    assert analysis.aggregate.ap_medium is None
    assert analysis.aggregate.ap_large is None
    assert analysis.operating_point.true_positives == 1
    assert analysis.operating_point.false_positives == 0
    assert analysis.operating_point.false_negatives == 0
    assert analysis.operating_point.f1 == pytest.approx(1.0)
    assert analysis.precision_recall_curve is not None
    assert analysis.per_class[0].precision_recall_curve is not None
    assert analysis.per_class[0].operating_point == analysis.operating_point


def test_coco_metrics_empty_predictions_are_valid_and_do_not_print(capsys):
    analysis = _evaluate(predictions=())

    assert analysis.aggregate.map_50_95 == pytest.approx(0.0)
    assert analysis.aggregate.ar_100 == pytest.approx(0.0)
    assert analysis.operating_point.true_positives == 0
    assert analysis.operating_point.false_positives == 0
    assert analysis.operating_point.false_negatives == 1
    assert analysis.operating_point.precision is None
    assert analysis.operating_point.recall == 0.0
    assert analysis.operating_point.f1 is None
    assert capsys.readouterr().out == ""


def test_coco_metrics_completely_incorrect_prediction():
    analysis = _evaluate(predictions=(_prediction((100, 100, 120, 120)),))

    assert analysis.aggregate.map_50_95 == pytest.approx(0.0)
    assert analysis.operating_point.true_positives == 0
    assert analysis.operating_point.false_positives == 1
    assert analysis.operating_point.false_negatives == 1
    assert analysis.operating_point.precision == 0.0
    assert analysis.operating_point.recall == 0.0
    assert analysis.operating_point.f1 is None


def test_coco_metrics_iou_can_separate_ap50_from_ap75():
    analysis = _evaluate(
        ground_truths=(_ground_truth((10, 10, 30, 30)),),
        predictions=(_prediction((14, 10, 34, 30)),),
    )

    assert analysis.aggregate.ap_50 == pytest.approx(1.0)
    assert analysis.aggregate.ap_75 == pytest.approx(0.0)
    assert 0.0 < analysis.aggregate.map_50_95 < 1.0


def test_coco_metrics_respect_score_ranking_in_precision_recall_curve():
    analysis = _evaluate(
        predictions=(
            _prediction((100, 100, 120, 120), confidence=0.95),
            _prediction((10, 10, 30, 30), confidence=0.8),
        )
    )

    assert analysis.aggregate.map_50_95 == pytest.approx(0.5)
    assert analysis.precision_recall_curve is not None
    assert analysis.precision_recall_curve.points[-1].precision == pytest.approx(0.5)
    curve_mean = sum(
        point.precision for point in analysis.precision_recall_curve.points
    ) / len(analysis.precision_recall_curve.points)
    assert curve_mean == pytest.approx(analysis.aggregate.map_50_95)
    class_curve = analysis.per_class[0].precision_recall_curve
    assert class_curve is not None
    class_curve_mean = sum(point.precision for point in class_curve.points) / len(
        class_curve.points
    )
    assert class_curve_mean == pytest.approx(analysis.per_class[0].average_precision)


def test_coco_metrics_multiclass_ids_are_independent_from_coco_ids():
    classes = (
        _class(
            2,
            internal_id="category-cat",
            coco_category_id=700,
            name="cat",
        ),
        _class(
            9,
            internal_id="category-dog",
            coco_category_id=3,
            name="dog",
        ),
    )
    analysis = _evaluate(
        classes=classes,
        ground_truths=(
            _ground_truth((10, 10, 30, 30), model_class_id=2),
            _ground_truth(
                (50, 50, 80, 80),
                annotation_id=2,
                model_class_id=9,
            ),
        ),
        predictions=(
            _prediction((10, 10, 30, 30), model_class_id=2),
            _prediction((50, 50, 80, 80), model_class_id=9),
        ),
    )

    assert analysis.aggregate.map_50_95 == pytest.approx(1.0)
    assert [item.reference.model_class_id for item in analysis.per_class] == [2, 9]
    assert [item.reference.coco_category_id for item in analysis.per_class] == [
        700,
        3,
    ]


def test_coco_metrics_include_resolved_class_with_zero_support_as_null():
    classes = (
        _class(0),
        _class(
            7,
            internal_id="category-unused",
            coco_category_id=800,
            name="unused",
        ),
    )
    analysis = _evaluate(classes=classes)

    unsupported = analysis.per_class[1]
    assert unsupported.support == 0
    assert unsupported.average_precision is None
    assert unsupported.average_recall is None
    assert unsupported.precision_recall_curve is None
    assert unsupported.operating_point.true_positives == 0
    assert unsupported.operating_point.false_positives == 0
    assert unsupported.operating_point.false_negatives == 0
    assert unsupported.operating_point.precision is None
    assert unsupported.operating_point.recall is None
    assert unsupported.operating_point.f1 is None


def test_coco_metrics_report_small_medium_and_large_when_supported():
    images = (
        _image("small"),
        _image("medium"),
        _image("large"),
    )
    ground_truths = (
        _ground_truth(
            (0, 0, 10, 10),
            image_id="small",
            annotation_id=1,
        ),
        _ground_truth(
            (0, 0, 50, 50),
            image_id="medium",
            annotation_id=2,
        ),
        _ground_truth(
            (0, 0, 120, 120),
            image_id="large",
            annotation_id=3,
        ),
    )
    predictions = tuple(
        _prediction(
            (
                ground_truth.box.x_min,
                ground_truth.box.y_min,
                ground_truth.box.x_max,
                ground_truth.box.y_max,
            ),
            image_id=ground_truth.image_id,
        )
        for ground_truth in ground_truths
    )

    analysis = _evaluate(
        images=images,
        ground_truths=ground_truths,
        predictions=predictions,
    )

    for value in (
        analysis.aggregate.ap_small,
        analysis.aggregate.ap_medium,
        analysis.aggregate.ap_large,
        analysis.aggregate.ar_small,
        analysis.aggregate.ar_medium,
        analysis.aggregate.ar_large,
    ):
        assert value == pytest.approx(1.0)


def test_coco_metrics_crowd_objects_and_matching_detections_are_ignored():
    analysis = _evaluate(
        ground_truths=(_ground_truth((10, 10, 30, 30), iscrowd=1),),
        predictions=(_prediction((10, 10, 30, 30)),),
    )

    assert analysis.aggregate.map_50_95 is None
    assert analysis.per_class[0].support == 0
    assert analysis.per_class[0].iscrowd_objects == 1
    assert analysis.operating_point.true_positives == 0
    assert analysis.operating_point.false_positives == 0
    assert analysis.operating_point.false_negatives == 0
    assert analysis.per_class[0].operating_point == analysis.operating_point


def test_coco_metrics_input_order_does_not_change_output():
    classes = (
        _class(2, internal_id="a", coco_category_id=50, name="a"),
        _class(9, internal_id="b", coco_category_id=10, name="b"),
    )
    ground_truths = (
        _ground_truth((10, 10, 30, 30), model_class_id=2),
        _ground_truth(
            (50, 50, 80, 80),
            annotation_id=2,
            model_class_id=9,
        ),
    )
    predictions = (
        _prediction((10, 10, 30, 30), model_class_id=2),
        _prediction((50, 50, 80, 80), model_class_id=9),
    )

    first = _evaluate(
        classes=classes,
        ground_truths=ground_truths,
        predictions=predictions,
    )
    second = _evaluate(
        classes=tuple(reversed(classes)),
        ground_truths=tuple(reversed(ground_truths)),
        predictions=tuple(reversed(predictions)),
    )

    assert first == second


def test_coco_metrics_ar_respects_standard_max_detections():
    analysis = _evaluate(
        ground_truths=(
            _ground_truth((10, 10, 30, 30), annotation_id=1),
            _ground_truth((50, 50, 70, 70), annotation_id=2),
        ),
        predictions=(
            _prediction((10, 10, 30, 30), confidence=0.9),
            _prediction((50, 50, 70, 70), confidence=0.8),
        ),
    )

    assert analysis.aggregate.ar_1 == pytest.approx(0.5)
    assert analysis.aggregate.ar_10 == pytest.approx(1.0)
    assert analysis.aggregate.ar_100 == pytest.approx(1.0)


def test_coco_metrics_operating_point_uses_configured_confidence():
    analysis = _evaluate(
        predictions=(
            _prediction((10, 10, 30, 30), confidence=0.4),
            _prediction((100, 100, 120, 120), confidence=0.3),
        ),
        confidence_threshold=0.35,
    )

    assert analysis.operating_point.true_positives == 1
    assert analysis.operating_point.false_positives == 0
    assert analysis.operating_point.false_negatives == 0


def test_coco_metrics_operating_point_treats_low_score_match_as_false_negative():
    analysis = _evaluate(
        predictions=(_prediction((10, 10, 30, 30), confidence=0.2),),
        confidence_threshold=0.25,
    )

    assert analysis.aggregate.map_50_95 == pytest.approx(1.0)
    assert analysis.operating_point.true_positives == 0
    assert analysis.operating_point.false_positives == 0
    assert analysis.operating_point.false_negatives == 1
    assert analysis.operating_point.precision is None
    assert analysis.operating_point.recall == 0.0
    assert analysis.operating_point.f1 is None


def test_coco_metrics_xyxy_to_xywh_preserves_full_precision():
    box = (1.123456789, 2.234567891, 11.987654321, 14.876543219)
    ground_truth = _ground_truth(box)
    prediction = _prediction(box, confidence=0.912345678)

    coco_gt, coco_dt, _ = _build_coco_apis(
        images=(_image(),),
        classes=(_class(),),
        ground_truths=(ground_truth,),
        predictions=(prediction,),
    )

    expected = [
        box[0],
        box[1],
        box[2] - box[0],
        box[3] - box[1],
    ]
    assert coco_gt.dataset["annotations"][0]["bbox"] == expected
    assert coco_dt.dataset["annotations"][0]["bbox"] == expected
    assert coco_dt.dataset["annotations"][0]["score"] == 0.912345678


def test_coco_metrics_never_expose_nan_infinity_or_sentinel():
    analysis = _evaluate(predictions=())

    values = [
        getattr(analysis.aggregate, field_name)
        for field_name in analysis.aggregate.__dataclass_fields__
    ]
    values.extend(
        (
            analysis.operating_point.precision,
            analysis.operating_point.recall,
            analysis.operating_point.f1,
        )
    )
    for value in values:
        assert value != -1
        assert value is None or math.isfinite(value)

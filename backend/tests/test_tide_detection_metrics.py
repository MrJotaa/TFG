import pytest

from ml.metrics.coco_detection import (
    CocoEvaluationClass,
    CocoEvaluationImage,
    CocoGroundTruth,
    CocoPrediction,
)
from ml.metrics.tide_detection import (
    TIDE_MAIN_ERROR_ORDER,
    TIDE_SPECIAL_ERROR_ORDER,
    evaluate_tide_detections,
)
from ml.models.detection import Detection, DetectionBox


def _image(image_id="image-1"):
    return CocoEvaluationImage(image_id=image_id, width=200, height=200)


def _class(class_id, name):
    return CocoEvaluationClass(
        category_internal_id=f"category-{name}",
        coco_category_id=class_id + 100,
        category_name=name,
        model_class_id=class_id,
        model_class_name=name,
    )


def _ground_truth(
    box=(20, 20, 60, 60),
    *,
    annotation_id=1,
    image_id="image-1",
    class_id=0,
    iscrowd=0,
):
    return CocoGroundTruth(
        annotation_id=annotation_id,
        object_internal_id=f"object-{annotation_id}",
        image_id=image_id,
        model_class_id=class_id,
        box=DetectionBox(*box),
        area=(box[2] - box[0]) * (box[3] - box[1]),
        iscrowd=iscrowd,
    )


def _prediction(
    box=(20, 20, 60, 60),
    *,
    image_id="image-1",
    class_id=0,
    confidence=0.9,
):
    return CocoPrediction(
        image_id=image_id,
        detection=Detection(
            class_id=class_id,
            class_name=f"class-{class_id}",
            confidence=confidence,
            box=DetectionBox(*box),
        ),
    )


def _evaluate(*, predictions, images=None, ground_truths=None, name="run"):
    return evaluate_tide_detections(
        images=tuple(images or (_image(),)),
        classes=(_class(0, "person"), _class(1, "car")),
        ground_truths=tuple(
            ground_truths if ground_truths is not None else (_ground_truth(),)
        ),
        predictions=tuple(predictions),
        name=name,
    )


def _counts(analysis):
    return {item.code: item.count for item in analysis.main_errors}


def test_tide_perfect_detection_has_full_ap_and_no_errors():
    analysis = _evaluate(predictions=(_prediction(),))

    assert analysis.ap_50 == pytest.approx(100.0)
    assert tuple(item.code for item in analysis.main_errors) == (TIDE_MAIN_ERROR_ORDER)
    assert tuple(item.code for item in analysis.special_errors) == (
        TIDE_SPECIAL_ERROR_ORDER
    )
    assert all(item.count == 0 for item in analysis.main_errors)


@pytest.mark.parametrize(
    ("code", "predictions"),
    [
        ("Cls", (_prediction(class_id=1),)),
        ("Loc", (_prediction(box=(36, 20, 76, 60)),)),
        ("Both", (_prediction(box=(36, 20, 76, 60), class_id=1),)),
        ("Dupe", (_prediction(), _prediction(confidence=0.8))),
        ("Bkg", (_prediction(box=(120, 120, 150, 150)),)),
    ],
)
def test_tide_identifies_each_prediction_error_type(code, predictions):
    analysis = _evaluate(predictions=predictions, name=code)

    assert _counts(analysis)[code] == 1
    if code == "Cls":
        assert analysis.special_errors[1].count == 1


def test_tide_empty_predictions_are_valid_and_count_missed_objects():
    analysis = _evaluate(predictions=())

    assert analysis.ap_50 == 0.0
    assert _counts(analysis) == {
        "Cls": 0,
        "Loc": 0,
        "Both": 0,
        "Dupe": 0,
        "Bkg": 0,
        "Miss": 1,
    }
    assert analysis.special_errors[1].code == "FalseNeg"
    assert analysis.special_errors[1].count == 1


def test_tide_respects_crowds_and_reviewed_empty_images():
    images = (_image("object"), _image("empty"))
    ground_truths = (
        _ground_truth(image_id="object", annotation_id=1),
        _ground_truth(
            box=(100, 100, 160, 160),
            image_id="object",
            annotation_id=2,
            iscrowd=1,
        ),
    )
    analysis = _evaluate(
        images=images,
        ground_truths=ground_truths,
        predictions=(
            _prediction(image_id="object"),
            _prediction(
                box=(100, 100, 160, 160),
                image_id="object",
            ),
            _prediction(
                box=(20, 20, 50, 50),
                image_id="empty",
            ),
        ),
    )

    assert _counts(analysis)["Bkg"] == 1
    assert _counts(analysis)["Miss"] == 0

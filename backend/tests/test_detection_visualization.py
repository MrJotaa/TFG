from PIL import Image

from ml.models.detection import Detection, DetectionBox
from ml.models.detection_visualization import (
    detection_color,
    draw_detections_on_image,
)


def _detection(*, class_id: int, class_name: str | None) -> Detection:
    return Detection(
        class_id=class_id,
        class_name=class_name,
        confidence=0.91,
        box=DetectionBox(4, 6, 32, 30),
    )


def test_detection_color_is_stable_by_semantic_class_across_model_ids():
    yolo_person = _detection(class_id=0, class_name="person")
    rf_detr_person = _detection(class_id=1, class_name="Person")

    assert detection_color(yolo_person) == detection_color(rf_detr_person)


def test_detection_color_distinguishes_common_classes():
    assert detection_color(
        _detection(class_id=0, class_name="person")
    ) != detection_color(_detection(class_id=2, class_name="car"))


def test_detection_color_falls_back_to_class_id_without_a_name():
    assert detection_color(_detection(class_id=7, class_name=None)) == detection_color(
        _detection(class_id=7, class_name=None)
    )
    assert detection_color(_detection(class_id=7, class_name=None)) != detection_color(
        _detection(class_id=8, class_name=None)
    )


def test_draw_detections_keeps_source_geometry_and_uses_class_colors():
    image = Image.new("RGB", (64, 48), color=(10, 20, 30))
    person = _detection(class_id=0, class_name="person")
    car = Detection(
        class_id=2,
        class_name="car",
        confidence=0.82,
        box=DetectionBox(36, 8, 60, 34),
    )

    annotated = draw_detections_on_image(image, (person, car))

    assert annotated.size == image.size
    assert annotated.mode == "RGB"
    assert annotated.getpixel((4, 20)) == detection_color(person)
    assert annotated.getpixel((36, 20)) == detection_color(car)
    assert annotated.tobytes() != image.tobytes()

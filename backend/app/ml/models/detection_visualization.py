import colorsys
import hashlib
from collections.abc import Sequence

from PIL import Image, ImageDraw

from ml.models.detection import Detection


def _detection_color_key(detection: Detection) -> str:
    if detection.class_name is not None:
        return f"name:{detection.class_name.strip().casefold()}"
    return f"id:{detection.class_id}"


def detection_color(detection: Detection) -> tuple[int, int, int]:
    """Return a stable, model-independent color for one semantic class."""
    digest = hashlib.blake2b(
        _detection_color_key(detection).encode("utf-8"),
        digest_size=8,
        person=b"tfg-detect",
    ).digest()
    hue = int.from_bytes(digest[:2], "big") / 65535.0
    saturation = 0.65 + (digest[2] / 255.0) * 0.20
    value = 0.82 + (digest[3] / 255.0) * 0.16
    red, green, blue = colorsys.hsv_to_rgb(hue, saturation, value)
    return (round(red * 255), round(green * 255), round(blue * 255))


def draw_detections_on_image(
    image: Image.Image,
    detections: Sequence[Detection],
) -> Image.Image:
    """Draw canonical detections using stable colors derived from class names."""
    output_image = image.convert("RGB").copy()
    draw = ImageDraw.Draw(output_image)

    for detection in detections:
        box = detection.box
        color = detection_color(detection)
        draw.rectangle(
            [(box.x_min, box.y_min), (box.x_max, box.y_max)],
            outline=color,
            width=3,
        )

        label = detection.class_name or str(detection.class_id)
        text = f"{label} {detection.confidence:.2f}"
        text_x = max(0, box.x_min)
        text_y = max(0, box.y_min - 18)
        text_bbox = draw.textbbox((text_x, text_y), text)
        draw.rectangle(text_bbox, fill=color)
        draw.text((text_x, text_y), text, fill=(0, 0, 0))

    return output_image

"""Compatibilidad de importación para el antiguo nombre YOLO Confidence."""

from ml.attacks.detection_confidence import DetectionConfidenceAttack

# El ID persistido sigue siendo ``yolo_confidence`` para no invalidar clientes,
# URLs ni resultados históricos. La implementación ya es agnóstica al detector.
YOLOConfidenceAttack = DetectionConfidenceAttack

__all__ = ["DetectionConfidenceAttack", "YOLOConfidenceAttack"]

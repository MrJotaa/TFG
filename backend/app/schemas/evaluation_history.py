from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from schemas.attack import ArchivedAttackId, AttackParameters, AttackType
from schemas.detection_evaluation import DetectionEvaluationResponse
from schemas.detection_metrics import DetectionSupervisedConfiguration
from schemas.evaluation import EvaluationResponse

EvaluationTaskType = Literal["classification", "detection"]
EvaluationStatus = Literal["running", "completed"]


class EvaluationCompatibilityRequest(BaseModel):
    task_type: EvaluationTaskType
    dataset_id: str
    attack_type: AttackType
    parameters: AttackParameters = Field(default_factory=AttackParameters)
    supervised_configuration: DetectionSupervisedConfiguration | None = None


class EvaluationHistoryItem(BaseModel):
    result_id: str
    task_type: EvaluationTaskType
    status: EvaluationStatus
    dataset_id: str
    dataset_name: str
    dataset_revision: int = Field(ge=0)
    model_id: str
    model_name: str
    attack_type: ArchivedAttackId
    parameters: AttackParameters
    supervised_configuration: DetectionSupervisedConfiguration | None = None
    summary: dict[str, Any] | None = None
    created_at: datetime
    completed_at: datetime | None = None
    expires_at: datetime | None = None
    retention_seconds: int | None = None
    preserved: bool = False
    artifact_available: bool = True
    exact_reuse_available: bool = False


class EvaluationHistoryPage(BaseModel):
    items: list[EvaluationHistoryItem]
    total: int = Field(ge=0)
    limit: int = Field(ge=1)
    offset: int = Field(ge=0)


class EvaluationCompatibilityResponse(BaseModel):
    match: EvaluationHistoryItem | None = None


class StoredEvaluationResponse(BaseModel):
    task_type: EvaluationTaskType
    evaluation: EvaluationResponse | DetectionEvaluationResponse


class EvaluationPreservationUpdate(BaseModel):
    preserved: bool

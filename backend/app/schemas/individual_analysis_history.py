from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from schemas.attack import AttackParameters, AttackResponse, AttackType
from schemas.detection_attack import DetectionAttackResponse

IndividualAnalysisTaskType = Literal["classification", "detection"]


class IndividualAnalysisSaveRequest(BaseModel):
    task_type: IndividualAnalysisTaskType
    result_id: str = Field(min_length=1)


class IndividualAnalysisCompatibilityRequest(BaseModel):
    task_type: IndividualAnalysisTaskType
    dataset_id: str
    image_id: str
    attack_type: AttackType
    parameters: AttackParameters = Field(default_factory=AttackParameters)


class IndividualAnalysisHistoryItem(BaseModel):
    analysis_id: str
    task_type: IndividualAnalysisTaskType
    dataset_id: str
    dataset_name: str
    dataset_revision: int = Field(ge=0)
    image_id: str
    image_filename: str
    model_id: str
    model_name: str
    attack_type: AttackType
    parameters: AttackParameters
    summary: dict[str, Any]
    explanation_available: bool = False
    created_at: datetime
    saved_at: datetime


class IndividualAnalysisHistoryPage(BaseModel):
    items: list[IndividualAnalysisHistoryItem]
    total: int = Field(ge=0)
    limit: int = Field(ge=1)
    offset: int = Field(ge=0)


class IndividualAnalysisCompatibilityResponse(BaseModel):
    match: IndividualAnalysisHistoryItem | None = None


class StoredIndividualAnalysisResponse(BaseModel):
    item: IndividualAnalysisHistoryItem
    analysis: AttackResponse | DetectionAttackResponse

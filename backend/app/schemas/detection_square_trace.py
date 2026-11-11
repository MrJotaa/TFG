"""Telemetría operativa opcional de Square-style Detection.

No sustituye métricas COCO ni el indicador histórico ``attack_success``.
El conteo comprende sólo las inferencias internas de ``SquareDetectionQuerySession``;
las inferencias posteriores del servicio quedan fuera de este presupuesto.
"""

from typing import Literal

from pydantic import BaseModel, Field, model_validator


SquareExecutionStatus = Literal[
    "completed",
    "budget_exhausted",
    "skipped_no_references",
    "skipped_zero_epsilon",
    "skipped_insufficient_budget",
]


class SquareDetectionExecutionTrace(BaseModel):
    status: SquareExecutionStatus
    queries: int = Field(ge=0)
    max_queries: int = Field(ge=1)
    cache_hits: int = Field(ge=0)
    seed: int = Field(ge=0)
    attack_revision: str = Field(min_length=1)
    reference_count: int = Field(ge=0)
    clean_score: float | None = Field(default=None, ge=0.0, le=1.0)
    best_score: float | None = Field(default=None, ge=0.0, le=1.0)
    objective_reached: bool = False

    @model_validator(mode="after")
    def validate_consistency(self):
        if self.queries > self.max_queries:
            raise ValueError("Square query count exceeds the budget")
        if self.reference_count == 0:
            if self.clean_score is not None or self.best_score is not None:
                raise ValueError("No references: survival scores must be absent")
            if self.objective_reached:
                raise ValueError("Cannot reach an objective without references")
        else:
            if self.clean_score is None or self.best_score is None:
                raise ValueError("Reference scores are required")
            if self.best_score > self.clean_score + 1e-6:
                raise ValueError("Best score cannot exceed the clean score")
        if self.status == "budget_exhausted" and self.queries != self.max_queries:
            raise ValueError("Budget exhaustion requires using all queries")
        return self

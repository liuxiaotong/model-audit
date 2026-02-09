"""数据模型 — Fingerprint / ComparisonResult / DetectionResult / AuditResult."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class Fingerprint(BaseModel):
    """模型指纹."""

    model_id: str
    method: str
    fingerprint_type: Literal["whitebox", "blackbox"]
    data: dict[str, Any]
    created_at: datetime = Field(default_factory=datetime.now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ComparisonResult(BaseModel):
    """指纹比对结果."""

    model_a: str
    model_b: str
    method: str
    similarity: float = Field(ge=0.0, le=1.0)
    is_derived: bool
    threshold: float
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    details: dict[str, Any] = Field(default_factory=dict)


class DetectionResult(BaseModel):
    """文本来源检测结果."""

    text_id: str | int
    text_preview: str = ""
    predicted_model: str
    confidence: float = Field(ge=0.0, le=1.0)
    scores: dict[str, float] = Field(default_factory=dict)


class AuditResult(BaseModel):
    """审计报告."""

    model_a: str
    model_b: str
    comparisons: list[ComparisonResult] = Field(default_factory=list)
    verdict: Literal["likely_derived", "independent", "inconclusive"] = "inconclusive"
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    summary: str = ""
    details: dict[str, Any] = Field(default_factory=dict)

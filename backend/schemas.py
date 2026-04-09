from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AnalyzeResponse(BaseModel):
    type: str = Field(default="unknown", description="Detected content type.")
    signal_score: float = Field(default=0.5, ge=0.0, le=1.0, description="Manipulation concern score.")
    confidence: float = Field(default=0.4, ge=0.0, le=1.0)
    status: str = Field(default="Suspicious", description="Authenticity analysis status.")
    explanation: str = ""
    evidence: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SavedPrompt(BaseModel):
    id: str
    messages: list[dict[str, Any]]


class SavedPrediction(BaseModel):
    id: str
    raw_response: str | None = None
    answer: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SavedEvaluation(BaseModel):
    id: str
    outcome: str
    metrics: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

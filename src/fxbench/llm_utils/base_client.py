from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ValidationError


class BaseLLMClient(ABC):
    def extract_json_from_response(self, response_content: str) -> str:
        if not response_content:
            return response_content

        stripped_content = response_content.strip()
        if stripped_content.startswith("```json") and stripped_content.endswith("```"):
            lines = stripped_content.split("\n")
            if len(lines) > 2:
                return "\n".join(lines[1:-1]).strip()

        if stripped_content.startswith("```") and stripped_content.endswith("```"):
            lines = stripped_content.split("\n")
            if len(lines) > 2:
                inner_content = "\n".join(lines[1:-1]).strip()
                if inner_content.startswith("{") and inner_content.endswith("}"):
                    return inner_content

        return stripped_content

    def validate_response(
        self,
        response_content: str,
        response_model: type[BaseModel],
    ) -> tuple[BaseModel, dict[str, Any]]:
        try:
            response_json = json.loads(response_content)
        except json.JSONDecodeError as error:
            raise ValueError(f"Failed to parse response as JSON: {error}") from error

        try:
            validated_response = response_model.model_validate(response_json)
        except ValidationError as error:
            raise ValueError(f"Response validation failed: {error}") from error

        return validated_response, response_json

    def count_tokens(self, text: str) -> int:
        return int(len(text.split()) * 1.3)

    @abstractmethod
    def prompt_llm(
        self,
        model: str | None,
        message_history: list[dict[str, Any]] | None = None,
        question: str | None = None,
        base64_images: list[str] | None = None,
        response_model: type[BaseModel] | None = None,
        max_tokens: int = 4000,
        temperature: float = 0.7,
        top_p: float = 0.95,
        retries: int = 20,
        retry_delay: int = 5,
    ) -> Any:
        raise NotImplementedError

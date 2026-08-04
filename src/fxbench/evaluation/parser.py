"""Response parsing: extract formula from LLM raw responses."""

import json
import re
from abc import ABC, abstractmethod
from typing import Any, Optional

from pydantic import BaseModel


class ParsedResult(BaseModel):
    """Result of parsing an LLM response."""
    formula: Optional[str] = None
    error: Optional[str] = None
    metadata: dict[str, Any] = {}


class IResponseParser(ABC):
    """Interface for parsing LLM responses into formula predictions."""

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def parse(self, raw_response: str) -> ParsedResult:
        raise NotImplementedError()


class JsonParser(IResponseParser):
    """Parses JSON responses to extract formula field."""

    def __init__(self, formula_key: str = "formula", intent_key: str = "intent"):
        self.formula_key = formula_key
        self.intent_key = intent_key

    def parse(self, raw_response: str) -> ParsedResult:
        text = raw_response.strip()

        # Try to extract JSON from markdown code blocks
        md_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
        if md_match:
            text = md_match.group(1).strip()

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return ParsedResult(error="JSON decode error", metadata={"raw": raw_response})

        formula = parsed.get(self.formula_key)
        intent = parsed.get(self.intent_key)

        if formula is not None:
            return ParsedResult(formula=str(formula), metadata={"intent": intent} if intent else {})
        return ParsedResult(error="No formula field in response", metadata={"parsed": parsed})

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from typing import Any

import tiktoken
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
from openai import AzureOpenAI
from pydantic import BaseModel

from .base_client import BaseLLMClient


load_dotenv()

ENDPOINT_URL = os.getenv("ENDPOINT_URL")
DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME")
MANAGED_CLIENT_ID = os.getenv("MANAGED_CLIENT_ID")
API_VERSION = os.getenv("API_VERSION")
COGNITIVE_SERVICES_SCOPE = "https://cognitiveservices.azure.com/.default"
REASONING_MODELS = {
    model.strip()
    for model in os.getenv("REASONING_MODELS_LIST", "").split(",")
    if model.strip()
}


class LLMClient(BaseLLMClient):
    def __init__(
        self,
        endpoint_url: str | None = ENDPOINT_URL,
        managed_client_id: str | None = MANAGED_CLIENT_ID,
        api_version: str | None = API_VERSION,
    ):
        self.endpoint_url = endpoint_url
        self.managed_client_id = managed_client_id
        self.api_version = api_version
        self.client = self.initialize_client()

    def token_provider(self):
        credential_kwargs = {}
        if self.managed_client_id:
            credential_kwargs["managed_identity_client_id"] = self.managed_client_id
        credential = DefaultAzureCredential(**credential_kwargs)

        def get_token() -> str:
            return credential.get_token(COGNITIVE_SERVICES_SCOPE).token

        return get_token

    def initialize_client(self) -> AzureOpenAI:
        if not self.endpoint_url:
            raise ValueError("ENDPOINT_URL is required")
        if not self.api_version:
            raise ValueError("API_VERSION is required")
        return AzureOpenAI(
            azure_endpoint=self.endpoint_url,
            azure_ad_token_provider=self.token_provider(),
            api_version=self.api_version,
        )

    def serialize_logprobs(self, logprobs_content) -> list[dict] | None:
        if not logprobs_content:
            return None

        serialized_content = []
        for token_logprob in logprobs_content:
            token_data = {
                "token": str(getattr(token_logprob, "token", "")),
                "logprob": float(getattr(token_logprob, "logprob", 0.0)),
                "bytes": getattr(token_logprob, "bytes", None),
                "top_logprobs": [],
            }
            for top_logprob in getattr(token_logprob, "top_logprobs", None) or []:
                token_data["top_logprobs"].append(
                    {
                        "token": str(getattr(top_logprob, "token", "")),
                        "logprob": float(getattr(top_logprob, "logprob", 0.0)),
                        "bytes": getattr(top_logprob, "bytes", None),
                    }
                )
            serialized_content.append(token_data)
        return serialized_content

    def metadata_for_completion(
        self,
        completion,
        request_start_time: datetime,
        request_end_time: datetime,
        messages: list[dict[str, Any]],
        temperature: float,
        top_p: float,
        max_tokens: int,
        retries_attempted: int,
        model: str,
    ) -> dict[str, Any]:
        choice = completion.choices[0]
        is_reasoning_model = model in REASONING_MODELS
        logprobs_requested = not is_reasoning_model
        logprobs_content = None

        if getattr(choice, "logprobs", None) is not None:
            content_logprobs = getattr(choice.logprobs, "content", None)
            logprobs_content = self.serialize_logprobs(content_logprobs)

        system_prompt_tokens = sum(
            self.count_tokens(str(message.get("content", "")))
            for message in messages
            if message.get("role") == "system"
        )

        return {
            "client_type": "azure_openai",
            "tokens": {
                "prompt_tokens": completion.usage.prompt_tokens if completion.usage else None,
                "completion_tokens": completion.usage.completion_tokens if completion.usage else None,
                "total_tokens": completion.usage.total_tokens if completion.usage else None,
                "available": completion.usage is not None,
            },
            "model_info": {
                "model_used": completion.model,
                "response_id": completion.id,
                "finish_reason": choice.finish_reason,
                "created": completion.created,
            },
            "timing": {
                "request_start": request_start_time.isoformat(),
                "response_end": request_end_time.isoformat(),
                "duration_ms": int((request_end_time - request_start_time).total_seconds() * 1000),
            },
            "request_config": {
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "retries_attempted": retries_attempted,
                "logprobs_enabled": logprobs_requested,
                "top_logprobs": 5 if logprobs_requested else None,
                "is_reasoning_model": is_reasoning_model,
            },
            "processing": {
                "system_prompt_tokens": system_prompt_tokens,
            },
            "logprobs": {
                "available": logprobs_content is not None,
                "requested": logprobs_requested,
                "content": logprobs_content,
                "reasoning_model": is_reasoning_model,
            },
        }

    def messages_for_prompt(
        self,
        message_history: list[dict[str, Any]] | None,
        question: str | None,
        base64_images: list[str] | None,
    ) -> list[dict[str, Any]]:
        if message_history:
            return message_history
        if question is None:
            raise ValueError("Either message_history or question must be provided")

        content = [{"type": "text", "text": question}]
        for base64_image in base64_images or []:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}",
                    },
                }
            )
        return [{"role": "user", "content": content}]

    def completion_params(
        self,
        model: str,
        messages: list[dict[str, Any]],
        response_model: type[BaseModel] | None,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> dict[str, Any]:
        params = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }

        if model in REASONING_MODELS:
            params.pop("max_tokens")
            params.pop("temperature")
            params.pop("top_p")
        else:
            params["logprobs"] = True
            params["top_logprobs"] = 5

        if response_model:
            params["response_format"] = {"type": "json_object"}

        return params

    def response_for_completion(
        self,
        completion,
        metadata: dict[str, Any],
        response_model: type[BaseModel] | None,
    ) -> dict[str, Any]:
        response_content = completion.choices[0].message.content or ""
        raw_response_length = len(response_content)
        clean_response = self.extract_json_from_response(response_content)

        if response_model:
            validated_response, response_json = self.validate_response(clean_response, response_model)
            metadata["processing"].update(
                {
                    "json_parse_success": True,
                    "raw_response_length": raw_response_length,
                    "extracted_answer_length": len(str(validated_response)),
                }
            )
            return {
                "validated_response": validated_response,
                "raw_json": response_json,
                "metadata": metadata,
            }

        extracted_answer = response_content
        json_parse_success = True
        try:
            response_json = json.loads(clean_response)
            if isinstance(response_json, dict):
                extracted_answer = response_json.get("answer", response_content)
        except (json.JSONDecodeError, TypeError):
            json_parse_success = False

        metadata["processing"].update(
            {
                "json_parse_success": json_parse_success,
                "raw_response_length": raw_response_length,
                "extracted_answer_length": len(extracted_answer) if extracted_answer else 0,
            }
        )
        return {
            "response": extracted_answer,
            "metadata": metadata,
        }

    def prompt_llm(
        self,
        model: str | None = None,
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
        selected_model = model or DEPLOYMENT_NAME
        if not selected_model:
            raise ValueError("model or DEPLOYMENT_NAME is required")

        messages = self.messages_for_prompt(message_history, question, base64_images)
        attempts = 0
        last_error = None

        while attempts <= retries:
            try:
                request_start_time = datetime.now()
                completion = self.client.chat.completions.create(
                    **self.completion_params(
                        selected_model,
                        messages,
                        response_model,
                        max_tokens,
                        temperature,
                        top_p,
                    )
                )
                request_end_time = datetime.now()
                metadata = self.metadata_for_completion(
                    completion,
                    request_start_time,
                    request_end_time,
                    messages,
                    temperature,
                    top_p,
                    max_tokens,
                    attempts,
                    selected_model,
                )
                return self.response_for_completion(completion, metadata, response_model)
            except Exception as error:
                last_error = error
                attempts += 1
                if attempts > retries:
                    raise last_error
                time.sleep(retry_delay)

        raise last_error

    def count_tokens(self, text: str) -> int:
        try:
            encoding = tiktoken.encoding_for_model("gpt-4")
            return len(encoding.encode(text))
        except Exception:
            return super().count_tokens(text)

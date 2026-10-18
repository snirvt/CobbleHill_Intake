"""Tests for constrained JSON output (classifier.providers.structured)."""

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from pydantic import BaseModel

from classifier.classifiers.patient_age import PatientAgeClassifier, PatientAgeOutput
from classifier.models import DocumentMetadata, PatientMetadata
from classifier.providers.ollama import SemaphoreChatModel, create_ollama_chat_model
from classifier.providers.structured import (
    bind_output_schema,
    ollama_schema_kwargs,
    vllm_schema_kwargs,
)
from classifier.providers.vllm import create_vllm_chat_model


class _Out(BaseModel):
    answer: str


def test_ollama_schema_kwargs_sends_schema_as_format() -> None:
    assert ollama_schema_kwargs(_Out) == {"format": _Out.model_json_schema()}


def test_vllm_schema_kwargs_sends_json_schema_response_format() -> None:
    assert vllm_schema_kwargs(_Out) == {
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "_Out", "schema": _Out.model_json_schema()},
        }
    }


def test_bind_output_schema_binds_provider_kwargs() -> None:
    llm = SemaphoreChatModel(
        inner=FakeListChatModel(responses=["{}"]),
        semaphore=asyncio.Semaphore(1),
        schema_kwargs=vllm_schema_kwargs,
    )
    bound = bind_output_schema(llm, _Out)
    assert bound.kwargs == vllm_schema_kwargs(_Out)  # type: ignore[attr-defined]


def test_bind_output_schema_passes_through_models_without_support() -> None:
    llm = FakeListChatModel(responses=["{}"])
    assert bind_output_schema(llm, _Out) is llm


def test_bind_output_schema_passes_through_wrapper_without_schema_kwargs() -> None:
    llm = SemaphoreChatModel(
        inner=FakeListChatModel(responses=["{}"]), semaphore=asyncio.Semaphore(1)
    )
    assert bind_output_schema(llm, _Out) is llm


def test_provider_factories_set_their_schema_kwargs() -> None:
    sem = asyncio.Semaphore(1)
    ollama = create_ollama_chat_model("http://localhost:11434", "m", semaphore=sem)
    vllm = create_vllm_chat_model(
        "http://localhost:8001", "m", temperature=0.0, top_p=1.0, semaphore=sem
    )
    assert isinstance(ollama, SemaphoreChatModel) and isinstance(vllm, SemaphoreChatModel)
    assert ollama.schema_kwargs is ollama_schema_kwargs
    assert vllm.schema_kwargs is vllm_schema_kwargs


@pytest.mark.asyncio
async def test_classifier_chain_sends_schema_to_inner_model() -> None:
    seen: list[dict[str, Any]] = []

    class RecordingModel(FakeListChatModel):
        async def _agenerate(self, messages: list[BaseMessage], **kwargs: Any) -> ChatResult:
            seen.append(kwargs)
            return await super()._agenerate(messages, **kwargs)

    llm = SemaphoreChatModel(
        inner=RecordingModel(responses=[json.dumps({"age_years": 72})]),
        semaphore=asyncio.Semaphore(1),
        schema_kwargs=vllm_schema_kwargs,
    )
    classifier = PatientAgeClassifier(llm=llm)

    result = await classifier.extract_age(
        DocumentMetadata(
            file_path=Path("dr_note.pdf"), raw_text="72 y.o. male", meta=PatientMetadata()
        )
    )

    assert result.age_years == 72
    assert seen[0]["response_format"] == vllm_schema_kwargs(PatientAgeOutput)["response_format"]

import json
from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from classifier.classifiers.treatment_request import TreatmentRequestClassifier
from classifier.models import DocumentMetadata, PatientMetadata
from classifier.providers.stub import (
    STUB_TREATMENT_REQUEST_RESPONSE,
    create_stub_chat_model,
)


def _fake_llm(requested: bool, reasoning: str = "ok") -> FakeListChatModel:
    return FakeListChatModel(
        responses=[
            json.dumps(
                {"treatment_requested": requested, "reasoning": reasoning}
            )
        ]
    )


def _make_metadata(raw_text: str = "Patient requested antibiotics.") -> DocumentMetadata:
    return DocumentMetadata(
        file_path=Path("dr_note.pdf"),
        raw_text=raw_text,
        meta=PatientMetadata(),
    )


@pytest.mark.asyncio
async def test_returns_true_with_reasoning() -> None:
    classifier = TreatmentRequestClassifier(llm=_fake_llm(True, "Patient asked for X"))

    result = await classifier.extract_treatment_request(_make_metadata())

    assert result.file_path == Path("dr_note.pdf")
    assert result.treatment_requested is True
    assert result.reasoning == "Patient asked for X"


@pytest.mark.asyncio
async def test_returns_false() -> None:
    classifier = TreatmentRequestClassifier(llm=_fake_llm(False, "No request found"))

    result = await classifier.extract_treatment_request(_make_metadata())

    assert result.treatment_requested is False
    assert result.reasoning == "No request found"


@pytest.mark.asyncio
async def test_malformed_response_defaults_to_false() -> None:
    classifier = TreatmentRequestClassifier(
        llm=FakeListChatModel(responses=["not valid json"])
    )

    result = await classifier.extract_treatment_request(_make_metadata())

    assert result.treatment_requested is False
    assert result.reasoning is not None and "error" in result.reasoning.lower()


@pytest.mark.asyncio
async def test_stub_provider_returns_deterministic_false() -> None:
    classifier = TreatmentRequestClassifier(
        llm=create_stub_chat_model(STUB_TREATMENT_REQUEST_RESPONSE)
    )

    result = await classifier.extract_treatment_request(_make_metadata())

    assert result.treatment_requested is False
    assert result.reasoning is not None and "STUB" in result.reasoning

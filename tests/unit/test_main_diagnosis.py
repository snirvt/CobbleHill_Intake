import json
from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from classifier.classifiers.main_diagnosis import MainDiagnosisClassifier
from classifier.models import DocumentMetadata, PatientMetadata
from classifier.providers.stub import (
    STUB_MAIN_DIAGNOSIS_RESPONSE,
    create_stub_chat_model,
)


def _fake_llm(label: str | None, reasoning: str = "ok") -> FakeListChatModel:
    return FakeListChatModel(
        responses=[json.dumps({"main_diagnosis": label, "reasoning": reasoning})]
    )


def _make_metadata(raw_text: str = "Assessment: community-acquired pneumonia.") -> DocumentMetadata:
    return DocumentMetadata(
        file_path=Path("dr_note.pdf"),
        raw_text=raw_text,
        meta=PatientMetadata(),
    )


# ---------------------------------------------------------------------------
# _normalize
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Pneumonia", "Pneumonia"),
        ("pneumonia", "Pneumonia"),
        ("PNEUMONIA", "Pneumonia"),
        ("otitis   media", "Otitis Media"),
        ('"Asthma."', "Asthma"),
        ("UTI", "UTI"),
        ("COPD exacerbation", "COPD Exacerbation"),
        ("Pneumonia (J18.9)", "Pneumonia"),
        ("Pneumonia J18.9", "Pneumonia"),
        ("Pneumonia (J18)", "Pneumonia"),
        ("Vitamin B12 deficiency", "Vitamin B12"),
        ("Acute otitis media", "Acute Otitis"),
        ("Crohn's disease", "Crohn's Disease"),
        ("COVID-19", "Covid-19"),
    ],
)
def test_normalize_produces_stable_grouping_label(raw: str, expected: str) -> None:
    assert MainDiagnosisClassifier._normalize(raw) == expected


@pytest.mark.parametrize(
    "raw", [None, "", "   ", "None", "n/a", "N/A", "null", "Unknown", "No diagnosis", "..."]
)
def test_normalize_returns_none_for_empty_labels(raw: str | None) -> None:
    assert MainDiagnosisClassifier._normalize(raw) is None


def test_normalize_truncates_to_configured_max_words(monkeypatch: pytest.MonkeyPatch) -> None:
    from config.settings import settings

    monkeypatch.setattr(settings, "main_diagnosis_max_words", 1)
    assert MainDiagnosisClassifier._normalize("Otitis media") == "Otitis"


# ---------------------------------------------------------------------------
# extract_main_diagnosis
# ---------------------------------------------------------------------------

async def test_returns_normalized_label_with_reasoning() -> None:
    classifier = MainDiagnosisClassifier(
        llm=_fake_llm("community-acquired pneumonia", "Assessment says CAP")
    )

    result = await classifier.extract_main_diagnosis(_make_metadata())

    assert result.file_path == Path("dr_note.pdf")
    assert result.main_diagnosis == "Community-acquired Pneumonia"
    assert result.reasoning == "Assessment says CAP"


async def test_null_label_returns_none() -> None:
    classifier = MainDiagnosisClassifier(llm=_fake_llm(None, "No diagnosis documented"))

    result = await classifier.extract_main_diagnosis(_make_metadata("Routine visit."))

    assert result.main_diagnosis is None
    assert result.reasoning == "No diagnosis documented"


async def test_malformed_response_returns_none_with_parse_error() -> None:
    classifier = MainDiagnosisClassifier(llm=FakeListChatModel(responses=["not valid json"]))

    result = await classifier.extract_main_diagnosis(_make_metadata())

    assert result.main_diagnosis is None
    assert result.reasoning is not None and result.reasoning.startswith("Parse error")


async def test_llm_failure_returns_none_with_llm_error() -> None:
    classifier = MainDiagnosisClassifier(llm=_fake_llm("Pneumonia"))

    async def boom(_input: object) -> object:
        raise TimeoutError("ollama timed out")

    classifier._chain = type("Chain", (), {"ainvoke": staticmethod(boom)})()  # type: ignore[assignment]

    result = await classifier.extract_main_diagnosis(_make_metadata())

    assert result.main_diagnosis is None
    assert result.reasoning is not None and result.reasoning.startswith("LLM error")


async def test_stub_provider_returns_obviously_fake_label() -> None:
    classifier = MainDiagnosisClassifier(llm=create_stub_chat_model(STUB_MAIN_DIAGNOSIS_RESPONSE))

    result = await classifier.extract_main_diagnosis(_make_metadata())

    assert result.main_diagnosis == "STUB Diagnosis"
    assert result.reasoning is not None and "STUB" in result.reasoning

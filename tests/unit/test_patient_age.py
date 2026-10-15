import json
from datetime import date
from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from classifier.classifiers.patient_age import PatientAgeClassifier, PatientAgeOutput
from classifier.models import DocumentMetadata, PatientMetadata
from classifier.providers.stub import STUB_PATIENT_AGE_RESPONSE, create_stub_chat_model


def _fake_llm(**fields: object) -> FakeListChatModel:
    return FakeListChatModel(responses=[json.dumps(fields)])


def _make_metadata() -> DocumentMetadata:
    return DocumentMetadata(
        file_path=Path("dr_note.pdf"), raw_text="Patient (8 mo F).", meta=PatientMetadata()
    )


# ---------------------------------------------------------------------------
# _compute_age
# ---------------------------------------------------------------------------

def test_compute_age_prefers_stated_age() -> None:
    output = PatientAgeOutput(age_years=0.67, dob="01/01/2000", date_of_service="01/01/2026")
    assert PatientAgeClassifier._compute_age(output) == 0.67


def test_compute_age_from_dob_and_date_of_service() -> None:
    output = PatientAgeOutput(dob="03/15/2010", date_of_service="03/15/2026")
    assert PatientAgeClassifier._compute_age(output) == pytest.approx(16.0, abs=0.01)


def test_compute_age_from_dob_uses_today_without_date_of_service() -> None:
    output = PatientAgeOutput(dob="2000-06-01")
    assert PatientAgeClassifier._compute_age(output, today=date(2025, 6, 1)) == pytest.approx(
        25.0, abs=0.01
    )


def test_compute_age_handles_two_digit_year() -> None:
    output = PatientAgeOutput(dob="01/01/20", date_of_service="01/01/26")
    assert PatientAgeClassifier._compute_age(output) == pytest.approx(6.0, abs=0.01)


@pytest.mark.parametrize(
    "output",
    [
        PatientAgeOutput(),
        PatientAgeOutput(dob="sometime in spring"),
        PatientAgeOutput(dob="01/01/2030", date_of_service="01/01/2026"),  # DOB after visit
        PatientAgeOutput(age_years=-1),
    ],
)
def test_compute_age_returns_none_when_undeterminable(output: PatientAgeOutput) -> None:
    assert PatientAgeClassifier._compute_age(output, today=date(2026, 1, 1)) is None


# ---------------------------------------------------------------------------
# extract_age
# ---------------------------------------------------------------------------

async def test_extract_age_returns_age_and_raw_fields() -> None:
    classifier = PatientAgeClassifier(
        llm=_fake_llm(
            age_years=0.67, dob="04/01/2025", date_of_service="12/01/2025", reasoning="8 mo"
        )
    )

    result = await classifier.extract_age(_make_metadata())

    assert result.file_path == Path("dr_note.pdf")
    assert result.age_years == 0.67
    assert result.dob == "04/01/2025"
    assert result.date_of_service == "12/01/2025"
    assert result.reasoning == "8 mo"


async def test_extract_age_malformed_response_returns_none() -> None:
    classifier = PatientAgeClassifier(llm=FakeListChatModel(responses=["not json"]))

    result = await classifier.extract_age(_make_metadata())

    assert result.age_years is None
    assert result.reasoning is not None and result.reasoning.startswith("Parse error")
    assert result.error is not None and result.error.startswith("Parse error")


async def test_extract_age_llm_failure_returns_none() -> None:
    classifier = PatientAgeClassifier(llm=_fake_llm(age_years=5))

    async def boom(_input: object) -> object:
        raise TimeoutError("ollama timed out")

    classifier._chain = type("Chain", (), {"ainvoke": staticmethod(boom)})()  # type: ignore[assignment]

    result = await classifier.extract_age(_make_metadata())

    assert result.age_years is None
    assert result.reasoning is not None and result.reasoning.startswith("LLM error")
    assert result.error == "LLM error: ollama timed out"


async def test_stub_provider_returns_obviously_fake_age() -> None:
    classifier = PatientAgeClassifier(llm=create_stub_chat_model(STUB_PATIENT_AGE_RESPONSE))

    result = await classifier.extract_age(_make_metadata())

    assert result.age_years == 99.0
    assert result.reasoning is not None and "STUB" in result.reasoning


async def test_extract_age_retries_malformed_response_then_succeeds() -> None:
    llm = FakeListChatModel(responses=["not json", json.dumps({"age_years": 4})])
    classifier = PatientAgeClassifier(llm=llm)

    result = await classifier.extract_age(_make_metadata())

    assert result.age_years == 4
    assert result.error is None


async def test_extract_age_success_has_no_error() -> None:
    classifier = PatientAgeClassifier(llm=_fake_llm(age_years=30))
    assert (await classifier.extract_age(_make_metadata())).error is None

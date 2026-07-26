import json
from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from classifier.classifiers.diagnosis_extraction import DiagnosisExtractionClassifier
from classifier.models import DocumentMetadata, PatientMetadata
from classifier.providers.stub import create_stub_chat_model


def _fake_llm(diagnoses: list[dict]) -> FakeListChatModel:
    """Return a FakeListChatModel emitting a JSON DiagnosisExtractionOutput."""
    return FakeListChatModel(responses=[json.dumps({"diagnoses": diagnoses})])


def _make_metadata(raw_text: str = "Assessment: Type 2 diabetes E11.9") -> DocumentMetadata:
    return DocumentMetadata(
        file_path=Path("dr_note.pdf"),
        raw_text=raw_text,
        meta=PatientMetadata(patient_name="Smith, John"),
    )


@pytest.mark.asyncio
async def test_extracts_diagnosis_with_icd_code() -> None:
    llm = _fake_llm([{"name": "Type 2 diabetes", "icd_code": "E11.9"}])
    classifier = DiagnosisExtractionClassifier(llm=llm)

    result = await classifier.extract_diagnoses(_make_metadata())

    assert result.file_path == Path("dr_note.pdf")
    assert len(result.diagnoses) == 1
    assert result.diagnoses[0].name == "Type 2 diabetes"
    assert result.diagnoses[0].icd_code == "E11.9"


@pytest.mark.asyncio
async def test_extracts_diagnosis_without_icd_code() -> None:
    llm = _fake_llm([{"name": "Hypertension", "icd_code": None}])
    classifier = DiagnosisExtractionClassifier(llm=llm)

    result = await classifier.extract_diagnoses(_make_metadata())

    assert len(result.diagnoses) == 1
    assert result.diagnoses[0].name == "Hypertension"
    assert result.diagnoses[0].icd_code is None


@pytest.mark.asyncio
async def test_extracts_multiple_diagnoses() -> None:
    llm = _fake_llm(
        [
            {"name": "Type 2 diabetes", "icd_code": "E11.9"},
            {"name": "Hypertension", "icd_code": None},
        ]
    )
    classifier = DiagnosisExtractionClassifier(llm=llm)

    result = await classifier.extract_diagnoses(_make_metadata())

    assert [d.name for d in result.diagnoses] == ["Type 2 diabetes", "Hypertension"]


@pytest.mark.asyncio
async def test_empty_result_when_no_diagnoses() -> None:
    llm = _fake_llm([])
    classifier = DiagnosisExtractionClassifier(llm=llm)

    result = await classifier.extract_diagnoses(_make_metadata())

    assert result.diagnoses == []


@pytest.mark.asyncio
async def test_malformed_response_returns_empty_list() -> None:
    llm = FakeListChatModel(responses=["not valid json"])
    classifier = DiagnosisExtractionClassifier(llm=llm)

    result = await classifier.extract_diagnoses(_make_metadata())

    assert result.file_path == Path("dr_note.pdf")
    assert result.diagnoses == []


@pytest.mark.asyncio
async def test_stub_provider_returns_fixed_fake_diagnosis() -> None:
    classifier = DiagnosisExtractionClassifier(llm=create_stub_chat_model())

    result = await classifier.extract_diagnoses(_make_metadata())

    assert len(result.diagnoses) == 1
    assert result.diagnoses[0].name == "STUB_DIAGNOSIS"
    assert result.diagnoses[0].icd_code == "Z00.0"


def test_create_stub_chat_model_accepts_custom_response() -> None:
    custom = json.dumps({"diagnoses": [{"name": "X", "icd_code": None}]})
    llm = create_stub_chat_model(custom)

    assert llm.responses == [custom]

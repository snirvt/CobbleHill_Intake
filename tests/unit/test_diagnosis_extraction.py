import json
from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from classifier.classifiers.diagnosis_extraction import (
    _PROMPT_TEMPLATE,
    DiagnosisExtractionClassifier,
)
from classifier.models import DiagnosisCategory, DocumentMetadata, PatientMetadata
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
    assert result.diagnosis_categories == []


@pytest.mark.asyncio
async def test_each_diagnosis_gets_its_category() -> None:
    llm = _fake_llm(
        [
            {"name": "Pneumonia", "icd_code": "J18.9", "category": "Respiratory"},
            {"name": "Hypertension", "icd_code": None, "category": "Cardiovascular"},
        ]
    )
    classifier = DiagnosisExtractionClassifier(llm=llm)

    result = await classifier.extract_diagnoses(_make_metadata())

    assert [d.category for d in result.diagnoses] == [
        DiagnosisCategory.RESPIRATORY,
        DiagnosisCategory.CARDIOVASCULAR,
    ]


@pytest.mark.asyncio
async def test_diagnosis_categories_are_deduped_in_order() -> None:
    llm = _fake_llm(
        [
            {"name": "Pneumonia", "icd_code": None, "category": "Respiratory"},
            {"name": "GERD", "icd_code": None, "category": "Reflux / GERD"},
            {"name": "Asthma", "icd_code": None, "category": "Respiratory"},
        ]
    )
    classifier = DiagnosisExtractionClassifier(llm=llm)

    result = await classifier.extract_diagnoses(_make_metadata())

    assert result.diagnosis_categories == [
        DiagnosisCategory.RESPIRATORY,
        DiagnosisCategory.REFLUX,
    ]


@pytest.mark.asyncio
async def test_unknown_category_is_parse_error() -> None:
    llm = _fake_llm([{"name": "Pneumonia", "icd_code": None, "category": "Lungs"}])
    classifier = DiagnosisExtractionClassifier(llm=llm)

    result = await classifier.extract_diagnoses(_make_metadata())

    assert result.diagnoses == []
    assert result.diagnosis_categories == []
    assert result.error is not None and result.error.startswith("Parse error")


@pytest.mark.asyncio
async def test_rendered_prompt_lists_categories_and_rules() -> None:
    llm = _fake_llm([])
    classifier = DiagnosisExtractionClassifier(llm=llm)

    seen: list[str] = []
    original = llm._call

    def spy(messages, *args, **kwargs):  # type: ignore[no-untyped-def]
        seen.append(str(messages[0].content))
        return original(messages, *args, **kwargs)

    object.__setattr__(llm, "_call", spy)
    await classifier.extract_diagnoses(_make_metadata())

    assert seen and all(f"- {c.value}" in seen[0] for c in DiagnosisCategory)
    assert "Pick the most specific category" in seen[0]


@pytest.mark.asyncio
async def test_reasoning_is_propagated_to_result() -> None:
    llm = FakeListChatModel(
        responses=[
            json.dumps(
                {
                    "reasoning": "Assessment lists type 2 diabetes with E11.9.",
                    "diagnoses": [{"name": "Type 2 diabetes", "icd_code": "E11.9"}],
                }
            )
        ]
    )
    classifier = DiagnosisExtractionClassifier(llm=llm)

    result = await classifier.extract_diagnoses(_make_metadata())

    assert result.reasoning == "Assessment lists type 2 diabetes with E11.9."
    assert result.error is None


@pytest.mark.asyncio
async def test_missing_diagnoses_key_sets_parse_error() -> None:
    llm = FakeListChatModel(responses=[json.dumps({"reasoning": "No list given."})])
    classifier = DiagnosisExtractionClassifier(llm=llm)

    result = await classifier.extract_diagnoses(_make_metadata())

    assert result.diagnoses == []
    assert result.error is not None and result.error.startswith("Parse error")


def test_prompt_prioritizes_assessment_and_falls_back_to_hpi() -> None:
    assert "Assessment" in _PROMPT_TEMPLATE
    assert "HPI or chief complaint" in _PROMPT_TEMPLATE


@pytest.mark.asyncio
async def test_malformed_response_returns_empty_list() -> None:
    llm = FakeListChatModel(responses=["not valid json"])
    classifier = DiagnosisExtractionClassifier(llm=llm)

    result = await classifier.extract_diagnoses(_make_metadata())

    assert result.file_path == Path("dr_note.pdf")
    assert result.diagnoses == []
    assert result.error is not None and result.error.startswith("Parse error")


@pytest.mark.asyncio
async def test_llm_failure_sets_error() -> None:
    classifier = DiagnosisExtractionClassifier(llm=FakeListChatModel(responses=["x"]))

    async def boom(_input: object) -> object:
        raise TimeoutError("ollama timed out")

    classifier._chain = type("Chain", (), {"ainvoke": staticmethod(boom)})()  # type: ignore[assignment]

    result = await classifier.extract_diagnoses(_make_metadata())

    assert result.diagnoses == []
    assert result.error == "LLM error: ollama timed out"


@pytest.mark.asyncio
async def test_stub_provider_returns_fixed_fake_diagnosis() -> None:
    classifier = DiagnosisExtractionClassifier(llm=create_stub_chat_model())

    result = await classifier.extract_diagnoses(_make_metadata())

    assert len(result.diagnoses) == 1
    assert result.diagnoses[0].name == "STUB_DIAGNOSIS"
    assert result.diagnoses[0].icd_code == "Z00.0"
    assert result.diagnosis_categories == [DiagnosisCategory.OTHER]


def test_create_stub_chat_model_accepts_custom_response() -> None:
    custom = json.dumps({"diagnoses": [{"name": "X", "icd_code": None}]})
    llm = create_stub_chat_model(custom)

    assert llm.responses == [custom]

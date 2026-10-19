import json
from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from classifier.classifiers.main_diagnosis import MainDiagnosisClassifier
from classifier.models import DiagnosisCategory, DocumentMetadata, PatientMetadata
from classifier.providers.stub import (
    STUB_MAIN_DIAGNOSIS_RESPONSE,
    create_stub_chat_model,
)


def _fake_llm(
    label: str | None,
    reasoning: str = "ok",
    category: str | None = None,
    icd_code: str | None = None,
) -> FakeListChatModel:
    return FakeListChatModel(
        responses=[
            json.dumps(
                {
                    "main_diagnosis": label,
                    "diagnosis_category": category,
                    "icd_code": icd_code,
                    "reasoning": reasoning,
                }
            )
        ]
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
        ("Vitamin B12 deficiency", "Vitamin B12 Deficiency"),
        ("Acute otitis media", "Acute Otitis Media"),
        ("Failure to thrive", "Failure To Thrive"),
        ("Acute on chronic systolic heart failure", "Acute On Chronic Systolic"),
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
    assert result.error is not None and result.error.startswith("Parse error")


async def test_llm_failure_returns_none_with_llm_error() -> None:
    classifier = MainDiagnosisClassifier(llm=_fake_llm("Pneumonia"))

    async def boom(_input: object) -> object:
        raise TimeoutError("ollama timed out")

    classifier._chain = type("Chain", (), {"ainvoke": staticmethod(boom)})()  # type: ignore[assignment]

    result = await classifier.extract_main_diagnosis(_make_metadata())

    assert result.main_diagnosis is None
    assert result.reasoning is not None and result.reasoning.startswith("LLM error")
    assert result.error == "LLM error: ollama timed out"


async def test_stub_provider_returns_obviously_fake_label() -> None:
    classifier = MainDiagnosisClassifier(llm=create_stub_chat_model(STUB_MAIN_DIAGNOSIS_RESPONSE))

    result = await classifier.extract_main_diagnosis(_make_metadata())

    assert result.main_diagnosis == "STUB Diagnosis"
    assert result.reasoning is not None and "STUB" in result.reasoning


def test_prompt_prefers_clinician_diagnosis_over_complaint() -> None:
    from classifier.classifiers.main_diagnosis import _PROMPT_TEMPLATE

    prompt = _PROMPT_TEMPLATE.lower()
    assert "assessment" in prompt and "impression" in prompt
    assert "clinician's diagnosis always wins over the patient's or family's complaint" in prompt
    # Fallback to the complaint only exists when there is no clinician diagnosis.
    assert prompt.index("assessment") < prompt.index("chief complaint")


def test_prompt_asks_for_complete_label_not_cut_off_phrase() -> None:
    from classifier.classifiers.main_diagnosis import _PROMPT_TEMPLATE

    prompt = _PROMPT_TEMPLATE.lower()
    # A hard "1-3 words" cap made the model chop phrases ("Encounter for ..." → "Encounter for").
    assert "1-3 words" not in prompt
    assert "never by cutting a phrase off" in prompt
    assert "never end on a connector word" in prompt
    # Encounter wording maps to the underlying condition.
    assert '"encounter for ..."' in prompt
    assert "not the encounter phrase" in prompt


# ---------------------------------------------------------------------------
# diagnosis_category
# ---------------------------------------------------------------------------

async def test_returns_category_alongside_label() -> None:
    classifier = MainDiagnosisClassifier(llm=_fake_llm("GERD", category="Reflux / GERD"))

    result = await classifier.extract_main_diagnosis(_make_metadata())

    assert result.main_diagnosis == "GERD"
    assert result.diagnosis_category is DiagnosisCategory.REFLUX


async def test_null_category_stays_none() -> None:
    classifier = MainDiagnosisClassifier(llm=_fake_llm(None, category=None))

    result = await classifier.extract_main_diagnosis(_make_metadata("Routine visit."))

    assert result.diagnosis_category is None
    assert result.error is None


async def test_category_dropped_when_label_means_no_diagnosis() -> None:
    classifier = MainDiagnosisClassifier(llm=_fake_llm("Unknown", category="Other"))

    result = await classifier.extract_main_diagnosis(_make_metadata())

    assert result.main_diagnosis is None
    assert result.diagnosis_category is None


async def test_category_outside_enum_is_parse_error() -> None:
    classifier = MainDiagnosisClassifier(llm=_fake_llm("GERD", category="Digestive Stuff"))

    result = await classifier.extract_main_diagnosis(_make_metadata())

    assert result.diagnosis_category is None
    assert result.error is not None and result.error.startswith("Parse error")


async def test_stub_provider_returns_valid_category() -> None:
    classifier = MainDiagnosisClassifier(llm=create_stub_chat_model(STUB_MAIN_DIAGNOSIS_RESPONSE))

    result = await classifier.extract_main_diagnosis(_make_metadata())

    assert result.diagnosis_category is DiagnosisCategory.OTHER


def test_prompt_lists_every_category() -> None:
    from classifier.classifiers.main_diagnosis import _CATEGORY_LIST

    for category in DiagnosisCategory:
        assert f"- {category.value}" in _CATEGORY_LIST


async def test_rendered_prompt_contains_categories() -> None:
    llm = _fake_llm("GERD", category="Reflux / GERD")
    classifier = MainDiagnosisClassifier(llm=llm)
    seen: list[str] = []
    original = llm._call

    def spy(messages, *args, **kwargs):  # type: ignore[no-untyped-def]
        seen.append(str(messages[0].content))
        return original(messages, *args, **kwargs)

    object.__setattr__(llm, "_call", spy)
    await classifier.extract_main_diagnosis(_make_metadata())

    assert seen and all(c.value in seen[0] for c in DiagnosisCategory)


def test_output_schema_constrains_category_to_enum() -> None:
    from classifier.classifiers.main_diagnosis import MainDiagnosisOutput

    schema = MainDiagnosisOutput.model_json_schema()
    enum = schema["$defs"]["DiagnosisCategory"]["enum"]
    assert enum == [c.value for c in DiagnosisCategory]


# ---------------------------------------------------------------------------
# icd_code
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("K21.9", "K21.9"),
        ("r10.83", "R10.83"),
        ("J18", "J18"),
        (" K21.9 ", "K21.9"),
        ("K21. 9", "K21.9"),
        ("S72.001A", "S72.001A"),
    ],
)
def test_normalize_icd_keeps_valid_codes(raw: str, expected: str) -> None:
    assert MainDiagnosisClassifier._normalize_icd(raw) == expected


@pytest.mark.parametrize("raw", [None, "", "   ", "K21-9", "abc", "21.9", "U07.1", "K21.99999"])
def test_normalize_icd_drops_invalid_codes(raw: str | None) -> None:
    assert MainDiagnosisClassifier._normalize_icd(raw) is None


async def test_icd_code_passed_through() -> None:
    classifier = MainDiagnosisClassifier(
        llm=_fake_llm("GERD", category="Reflux / GERD", icd_code="k21.9")
    )

    result = await classifier.extract_main_diagnosis(_make_metadata())

    assert result.icd_code == "K21.9"


async def test_malformed_icd_code_dropped_label_kept() -> None:
    classifier = MainDiagnosisClassifier(
        llm=_fake_llm("GERD", category="Reflux / GERD", icd_code="see chart")
    )

    result = await classifier.extract_main_diagnosis(_make_metadata())

    assert result.icd_code is None
    assert result.main_diagnosis == "GERD"
    assert result.diagnosis_category is DiagnosisCategory.REFLUX
    assert result.error is None


async def test_icd_code_dropped_when_label_means_no_diagnosis() -> None:
    classifier = MainDiagnosisClassifier(llm=_fake_llm("Unknown", icd_code="Z00.0"))

    result = await classifier.extract_main_diagnosis(_make_metadata())

    assert result.icd_code is None


async def test_stub_provider_returns_icd_code() -> None:
    classifier = MainDiagnosisClassifier(llm=create_stub_chat_model(STUB_MAIN_DIAGNOSIS_RESPONSE))

    result = await classifier.extract_main_diagnosis(_make_metadata())

    assert result.icd_code == "Z00.0"


def test_prompt_forbids_guessing_icd_code() -> None:
    from classifier.classifiers.main_diagnosis import _PROMPT_TEMPLATE

    assert "Never guess or look up a code yourself" in _PROMPT_TEMPLATE

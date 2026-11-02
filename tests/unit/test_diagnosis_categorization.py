import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from classifier.classifiers.categories import CATEGORY_EXAMPLES
from classifier.classifiers.diagnosis_categorization import (
    CategorizedDiagnosisExtractor,
    DiagnosisCategorizationClassifier,
    _format_diagnoses,
)
from classifier.models import (
    Diagnosis,
    DiagnosisCategorizationResult,
    DiagnosisCategory,
    DiagnosisExtractionResult,
    DocumentMetadata,
    PatientMetadata,
)
from classifier.providers.stub import STUB_DIAGNOSIS_CATEGORIES_RESPONSE, create_stub_chat_model

_DIAGNOSES = [
    Diagnosis(name="Underweight", icd_code="R63.6"),
    Diagnosis(name="Neonatal agitation", icd_code="R68.12"),
]


def _fake_llm(categories: list[dict], reasoning: str | None = None) -> FakeListChatModel:
    """Return a FakeListChatModel emitting a JSON DiagnosisCategorizationOutput."""
    return FakeListChatModel(
        responses=[json.dumps({"reasoning": reasoning, "categories": categories})]
    )


def _spy_prompts(llm: FakeListChatModel) -> list[str]:
    seen: list[str] = []
    original = llm._call

    def spy(messages, *args, **kwargs):  # type: ignore[no-untyped-def]
        seen.append(str(messages[0].content))
        return original(messages, *args, **kwargs)

    object.__setattr__(llm, "_call", spy)
    return seen


# ---------------------------------------------------------------------------
# DiagnosisCategorizationClassifier
# ---------------------------------------------------------------------------


def test_format_diagnoses_numbers_lines_and_adds_icd_when_present() -> None:
    diagnoses = [Diagnosis(name="Underweight", icd_code="R63.6"), Diagnosis(name="Colic")]

    assert _format_diagnoses(diagnoses) == "1. Underweight (R63.6)\n2. Colic"


async def test_categorize_assigns_categories_by_position() -> None:
    llm = _fake_llm(
        [
            {"name": "Underweight", "reason": "r", "category": "Feeding / Growth"},
            {"name": "Neonatal agitation", "reason": "r", "category": "Fussiness / Irritability"},
        ],
        reasoning="R63.6 is a growth concern.",
    )

    result = await DiagnosisCategorizationClassifier(llm=llm).categorize(_DIAGNOSES)

    assert result.error is None
    assert [d.category for d in result.diagnoses] == [
        DiagnosisCategory.FEEDING,
        DiagnosisCategory.FUSSINESS,
    ]
    assert [(d.name, d.icd_code) for d in result.diagnoses] == [
        ("Underweight", "R63.6"),
        ("Neonatal agitation", "R68.12"),
    ]
    assert result.reasoning == "R63.6 is a growth concern.\nUnderweight: r\nNeonatal agitation: r"


async def test_categorize_sets_each_category_reason_by_position() -> None:
    llm = _fake_llm(
        [
            {"name": "Underweight", "reason": "infant weight", "category": "Feeding / Growth"},
            {
                "name": "Neonatal agitation",
                "reason": "fussy",
                "category": "Fussiness / Irritability",
            },
        ]
    )

    result = await DiagnosisCategorizationClassifier(llm=llm).categorize(_DIAGNOSES)

    assert [d.category_reason for d in result.diagnoses] == ["infant weight", "fussy"]


async def test_categorize_keeps_input_names_when_model_rewords_them() -> None:
    llm = _fake_llm([{"name": "Low weight", "reason": "r", "category": "Feeding / Growth"}])

    result = await DiagnosisCategorizationClassifier(llm=llm).categorize(_DIAGNOSES[:1])

    assert result.diagnoses[0].name == "Underweight"
    assert result.diagnoses[0].category is DiagnosisCategory.FEEDING


async def test_categorize_count_mismatch_sets_error_and_leaves_categories_none() -> None:
    llm = _fake_llm([{"name": "Underweight", "reason": "r", "category": "Feeding / Growth"}])

    result = await DiagnosisCategorizationClassifier(llm=llm).categorize(_DIAGNOSES)

    assert result.error == "Category count mismatch: expected 2, got 1"
    assert result.diagnoses == _DIAGNOSES
    assert all(d.category is None for d in result.diagnoses)
    assert all(d.category_reason is None for d in result.diagnoses)


async def test_categorize_missing_reason_is_parse_error() -> None:
    llm = FakeListChatModel(
        responses=[
            json.dumps({"categories": [{"name": "Underweight", "category": "Feeding / Growth"}]})
        ]
    )

    result = await DiagnosisCategorizationClassifier(llm=llm).categorize(_DIAGNOSES[:1])

    assert result.error is not None and result.error.startswith("Category parse error")
    assert result.diagnoses == _DIAGNOSES[:1]


async def test_categorize_reasoning_lists_each_item_reason() -> None:
    llm = _fake_llm(
        [
            {
                "name": "Underweight",
                "reason": "3-month-old, weight rule child branch",
                "category": "Feeding / Growth",
            }
        ]
    )

    result = await DiagnosisCategorizationClassifier(llm=llm).categorize(_DIAGNOSES[:1], "3 months")

    assert result.reasoning == "Underweight: 3-month-old, weight rule child branch"


async def test_categorize_unknown_category_is_parse_error() -> None:
    llm = _fake_llm([{"name": "Underweight", "reason": "r", "category": "Weight"}])

    result = await DiagnosisCategorizationClassifier(llm=llm).categorize(_DIAGNOSES[:1])

    assert result.error is not None and result.error.startswith("Category parse error")
    assert result.diagnoses == _DIAGNOSES[:1]


async def test_categorize_llm_failure_sets_error() -> None:
    classifier = DiagnosisCategorizationClassifier(llm=FakeListChatModel(responses=["x"]))
    classifier._chain = AsyncMock()  # type: ignore[assignment]
    classifier._chain.ainvoke.side_effect = TimeoutError("ollama timed out")

    result = await classifier.categorize(_DIAGNOSES)

    assert result.error == "Category LLM error: ollama timed out"
    assert result.diagnoses == _DIAGNOSES


async def test_categorize_empty_input_skips_llm() -> None:
    classifier = DiagnosisCategorizationClassifier(llm=FakeListChatModel(responses=["x"]))
    classifier._chain = AsyncMock()  # type: ignore[assignment]

    result = await classifier.categorize([])

    assert result == DiagnosisCategorizationResult()
    classifier._chain.ainvoke.assert_not_awaited()


async def test_categorize_prompt_lists_diagnoses_categories_and_rules() -> None:
    llm = _fake_llm(
        [
            {"name": "Underweight", "reason": "r", "category": "Feeding / Growth"},
            {"name": "Neonatal agitation", "reason": "r", "category": "Fussiness / Irritability"},
        ]
    )
    seen = _spy_prompts(llm)

    await DiagnosisCategorizationClassifier(llm=llm).categorize(_DIAGNOSES)

    assert seen
    assert "1. Underweight (R63.6)\n2. Neonatal agitation (R68.12)" in seen[0]
    assert all(f"- {c.value}" in seen[0] for c in DiagnosisCategory)
    assert "Pick the most specific category" in seen[0]
    assert CATEGORY_EXAMPLES in seen[0]
    assert "applies to every diagnosis" in seen[0]


@pytest.mark.parametrize(("age", "shown"), [("3 months", "3 months"), (None, "unknown")])
async def test_categorize_prompt_shows_patient_age(age: str | None, shown: str) -> None:
    llm = _fake_llm([{"name": "Underweight", "reason": "r", "category": "Feeding / Growth"}])
    seen = _spy_prompts(llm)

    await DiagnosisCategorizationClassifier(llm=llm).categorize(_DIAGNOSES[:1], age)

    assert f"Patient age: {shown}" in seen[0]


async def test_categorize_with_stub_response_tags_other() -> None:
    llm = create_stub_chat_model(STUB_DIAGNOSIS_CATEGORIES_RESPONSE)

    result = await DiagnosisCategorizationClassifier(llm=llm).categorize(
        [Diagnosis(name="STUB_DIAGNOSIS", icd_code="Z00.0")]
    )

    assert result.diagnoses[0].category is DiagnosisCategory.OTHER


# ---------------------------------------------------------------------------
# CategorizedDiagnosisExtractor
# ---------------------------------------------------------------------------


def _metadata() -> DocumentMetadata:
    return DocumentMetadata(
        file_path=Path("dr_note.pdf"), raw_text="Assessment: x", meta=PatientMetadata()
    )


def _extractor(result: DiagnosisExtractionResult) -> AsyncMock:
    extractor = AsyncMock()
    extractor.extract_diagnoses.return_value = result
    return extractor


def _categorizer(result: DiagnosisCategorizationResult) -> AsyncMock:
    categorizer = AsyncMock()
    categorizer.categorize.return_value = result
    return categorizer


async def test_combined_extractor_merges_categories_and_dedupes() -> None:
    extracted = DiagnosisExtractionResult(
        file_path=Path("dr_note.pdf"),
        diagnoses=[Diagnosis(name="A"), Diagnosis(name="B"), Diagnosis(name="C")],
        reasoning="from assessment",
    )
    categorized = DiagnosisCategorizationResult(
        diagnoses=[
            Diagnosis(name="A", category=DiagnosisCategory.FEEDING),
            Diagnosis(name="B", category=DiagnosisCategory.REFLUX),
            Diagnosis(name="C", category=DiagnosisCategory.FEEDING),
        ],
        reasoning="growth terms",
    )
    categorizer = _categorizer(categorized)
    combined = CategorizedDiagnosisExtractor(_extractor(extracted), categorizer)

    result = await combined.extract_diagnoses(_metadata(), "6 months")

    categorizer.categorize.assert_awaited_once_with(extracted.diagnoses, "6 months")
    assert result.file_path == Path("dr_note.pdf")
    assert result.diagnoses == categorized.diagnoses
    assert result.diagnosis_categories == [DiagnosisCategory.FEEDING, DiagnosisCategory.REFLUX]
    assert result.reasoning == "extraction: from assessment\ncategories: growth terms"
    assert result.error is None


async def test_combined_extractor_skips_categorizer_on_extraction_error() -> None:
    extracted = DiagnosisExtractionResult(file_path=Path("dr_note.pdf"), error="LLM error: x")
    categorizer = _categorizer(DiagnosisCategorizationResult())
    combined = CategorizedDiagnosisExtractor(_extractor(extracted), categorizer)

    result = await combined.extract_diagnoses(_metadata())

    assert result == extracted
    categorizer.categorize.assert_not_awaited()


async def test_combined_extractor_keeps_diagnoses_when_categorizer_fails() -> None:
    diagnoses = [Diagnosis(name="A")]
    extracted = DiagnosisExtractionResult(file_path=Path("dr_note.pdf"), diagnoses=diagnoses)
    categorized = DiagnosisCategorizationResult(
        diagnoses=diagnoses, error="Category LLM error: boom"
    )
    combined = CategorizedDiagnosisExtractor(_extractor(extracted), _categorizer(categorized))

    result = await combined.extract_diagnoses(_metadata())

    assert result.diagnoses == diagnoses
    assert result.diagnosis_categories == []
    assert result.error == "Category LLM error: boom"


@pytest.mark.parametrize("reasoning", [None, ""])
async def test_combined_extractor_reasoning_none_when_both_steps_silent(
    reasoning: str | None,
) -> None:
    extracted = DiagnosisExtractionResult(file_path=Path("dr_note.pdf"), reasoning=reasoning)
    combined = CategorizedDiagnosisExtractor(
        _extractor(extracted), _categorizer(DiagnosisCategorizationResult())
    )

    result = await combined.extract_diagnoses(_metadata())

    assert result.reasoning is None

import pytest
from pydantic import BaseModel, ValidationError

from classifier.classifiers.diagnosis_categorization import (
    CategorizedItem,
    DiagnosisCategorizationOutput,
)
from classifier.classifiers.diagnosis_extraction import DiagnosisExtractionOutput
from classifier.classifiers.dr_nurse_match import ClinicalMatchOutput
from classifier.classifiers.main_diagnosis import MainDiagnosisOutput
from classifier.classifiers.patient_age import PatientAgeOutput
from classifier.classifiers.reasoning_guard import find_reasoning_problem
from classifier.classifiers.treatment_request import TreatmentRequestOutput
from config.settings import settings

GOOD_REASONING = (
    "Assessment lists preterm birth and low birth weight; the Plan recommends VNS and HHA. "
    "Both grounds apply."
)

# Synthetic copy of a real failure: the model echoes its output instructions in a loop.
ECHO_LOOP = (
    "The patient needs care. The output must be a valid JSON object conforming to the "
    "specified schema. The reasoning should be a string citing the relevant wording. "
    "The schema has properties treatment_requested (boolean) and reasoning (anyOf string "
    "or null). The required field is treatment_requested."
)

REPEATED_SENTENCE = " ".join(["The patient clearly needs ongoing care and treatment."] * 3)


def test_good_reasoning_passes() -> None:
    assert find_reasoning_problem(GOOD_REASONING) is None


def test_empty_and_none_pass() -> None:
    assert find_reasoning_problem(None) is None
    assert find_reasoning_problem("") is None


def test_echoed_instructions_are_flagged() -> None:
    problem = find_reasoning_problem(ECHO_LOOP)

    assert problem is not None and "echoes output instructions" in problem
    assert "json" in problem and "schema" in problem


def test_leak_markers_are_case_insensitive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "reasoning_leak_markers", ["output must"])

    assert find_reasoning_problem("THE OUTPUT MUST be short.") is not None


def test_repeated_sentence_is_flagged() -> None:
    problem = find_reasoning_problem(REPEATED_SENTENCE)

    assert problem is not None and "repeats a sentence" in problem


def test_sentence_repeated_below_limit_passes() -> None:
    text = " ".join(["The patient clearly needs ongoing care and treatment."] * 2)

    assert find_reasoning_problem(text) is None


def test_short_repeated_sentences_are_ignored() -> None:
    assert find_reasoning_problem("Stable. Stable. Stable. Doing well.") is None


def test_too_long_is_flagged(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "reasoning_max_chars", 50)

    problem = find_reasoning_problem(GOOD_REASONING)

    assert problem is not None and "too long" in problem


def test_mostly_repeated_phrases_are_flagged() -> None:
    # Near-duplicate sentences (one word changes) defeat the exact-sentence check.
    text = " ".join(
        f"Patient {i} has preterm status and low birth weight requiring follow up care"
        for i in range(8)
    )

    problem = find_reasoning_problem(text)

    assert problem is not None and "mostly repetition" in problem


OUTPUT_MODELS: list[tuple[type[BaseModel], dict[str, object]]] = [
    (TreatmentRequestOutput, {"treatment_requested": True}),
    (ClinicalMatchOutput, {"verdict": "MATCH"}),
    (PatientAgeOutput, {}),
    (DiagnosisExtractionOutput, {"diagnoses": []}),
    (MainDiagnosisOutput, {}),
    (DiagnosisCategorizationOutput, {"categories": []}),
]


@pytest.mark.parametrize(("model", "fields"), OUTPUT_MODELS)
def test_output_models_reject_degenerate_reasoning(
    model: type[BaseModel], fields: dict[str, object]
) -> None:
    with pytest.raises(ValidationError, match="degenerate LLM output"):
        model.model_validate({**fields, "reasoning": ECHO_LOOP})


@pytest.mark.parametrize(("model", "fields"), OUTPUT_MODELS)
def test_output_models_accept_good_reasoning(
    model: type[BaseModel], fields: dict[str, object]
) -> None:
    assert model.model_validate({**fields, "reasoning": GOOD_REASONING}) is not None


def test_categorized_item_rejects_degenerate_reason() -> None:
    with pytest.raises(ValidationError, match="degenerate LLM output"):
        CategorizedItem.model_validate(
            {"name": "Asthma", "reason": REPEATED_SENTENCE, "category": "Other"}
        )

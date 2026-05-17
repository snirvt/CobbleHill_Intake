from unittest.mock import AsyncMock

import pytest

from classifier.classifiers.doctor_visit_needed import DoctorVisitNeededClassifier
from classifier.classifiers.dummy import DummyClassifier
from classifier.classifiers.llm_classifier import LLMClassifier
from classifier.models import ClassificationResult, DocumentMetadata


# ---------------------------------------------------------------------------
# DummyClassifier
# ---------------------------------------------------------------------------

async def test_dummy_classifier_returns_test_category(
    sample_document_metadata: DocumentMetadata,
) -> None:
    clf = DummyClassifier()
    result = await clf.classify(sample_document_metadata)

    assert isinstance(result, ClassificationResult)
    assert result.category == "TEST"
    assert result.file_path == sample_document_metadata.file_path
    assert result.metadata == sample_document_metadata


async def test_dummy_classifier_preserves_metadata(
    sample_document_metadata: DocumentMetadata,
) -> None:
    clf = DummyClassifier()
    result = await clf.classify(sample_document_metadata)
    assert result.metadata.meta.patient_name == "Test, Patient"


# ---------------------------------------------------------------------------
# DoctorVisitNeededClassifier
# ---------------------------------------------------------------------------

def _make_llm(response: str) -> AsyncMock:
    llm = AsyncMock()
    llm.complete = AsyncMock(return_value=response)
    return llm


async def test_doctor_visit_needed_returns_needs_visit(
    sample_document_metadata: DocumentMetadata,
) -> None:
    clf = DoctorVisitNeededClassifier(llm=_make_llm("NEEDS_VISIT"))
    result = await clf.run(sample_document_metadata)
    assert result == "NEEDS_VISIT"


async def test_doctor_visit_needed_returns_no_visit_needed(
    sample_document_metadata: DocumentMetadata,
) -> None:
    clf = DoctorVisitNeededClassifier(llm=_make_llm("NO_VISIT_NEEDED"))
    result = await clf.run(sample_document_metadata)
    assert result == "NO_VISIT_NEEDED"


async def test_doctor_visit_needed_defaults_on_ambiguous_response(
    sample_document_metadata: DocumentMetadata,
) -> None:
    clf = DoctorVisitNeededClassifier(llm=_make_llm("I'm not sure, maybe?"))
    result = await clf.run(sample_document_metadata)
    assert result == "NEEDS_VISIT"


async def test_doctor_visit_needed_category_embedded_in_prose(
    sample_document_metadata: DocumentMetadata,
) -> None:
    clf = DoctorVisitNeededClassifier(llm=_make_llm("Based on notes: NO_VISIT_NEEDED."))
    result = await clf.run(sample_document_metadata)
    assert result == "NO_VISIT_NEEDED"


async def test_doctor_visit_needed_prompt_includes_complaints(
    sample_document_metadata: DocumentMetadata,
) -> None:
    llm = _make_llm("NEEDS_VISIT")
    clf = DoctorVisitNeededClassifier(llm=llm)
    await clf.run(sample_document_metadata)

    called_prompt: str = llm.complete.call_args[0][0]
    assert "Well Visit" in called_prompt
    assert "Runny nose" in called_prompt


async def test_doctor_visit_needed_prompt_includes_assessment(
    sample_document_metadata: DocumentMetadata,
) -> None:
    llm = _make_llm("NEEDS_VISIT")
    clf = DoctorVisitNeededClassifier(llm=llm)
    await clf.run(sample_document_metadata)

    called_prompt: str = llm.complete.call_args[0][0]
    assert "Z00.129" in called_prompt


async def test_doctor_visit_needed_task_name() -> None:
    assert DoctorVisitNeededClassifier.task_name == "doctor_visit_needed"


# ---------------------------------------------------------------------------
# LLMClassifier orchestrator
# ---------------------------------------------------------------------------

async def test_llm_classifier_runs_all_tasks(
    sample_document_metadata: DocumentMetadata,
) -> None:
    task_a = AsyncMock()
    task_a.task_name = "task_a"
    task_a.run = AsyncMock(return_value="RESULT_A")

    task_b = AsyncMock()
    task_b.task_name = "task_b"
    task_b.run = AsyncMock(return_value="RESULT_B")

    clf = LLMClassifier(tasks=[task_a, task_b])
    result = await clf.classify(sample_document_metadata)

    assert result.task_results == {"task_a": "RESULT_A", "task_b": "RESULT_B"}
    task_a.run.assert_awaited_once_with(sample_document_metadata)
    task_b.run.assert_awaited_once_with(sample_document_metadata)


async def test_llm_classifier_primary_category_is_first_task(
    sample_document_metadata: DocumentMetadata,
) -> None:
    task = AsyncMock()
    task.task_name = "doctor_visit_needed"
    task.run = AsyncMock(return_value="NEEDS_VISIT")

    clf = LLMClassifier(tasks=[task])
    result = await clf.classify(sample_document_metadata)

    assert result.category == "NEEDS_VISIT"


async def test_llm_classifier_unknown_category_when_no_tasks(
    sample_document_metadata: DocumentMetadata,
) -> None:
    clf = LLMClassifier(tasks=[])
    result = await clf.classify(sample_document_metadata)
    assert result.category == "UNKNOWN"
    assert result.task_results == {}


async def test_llm_classifier_preserves_file_path(
    sample_document_metadata: DocumentMetadata,
) -> None:
    task = AsyncMock()
    task.task_name = "t"
    task.run = AsyncMock(return_value="X")

    clf = LLMClassifier(tasks=[task])
    result = await clf.classify(sample_document_metadata)
    assert result.file_path == sample_document_metadata.file_path

import json
from unittest.mock import AsyncMock

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from classifier.classifiers.doctor_visit_needed import DoctorVisitNeededClassifier
from classifier.classifiers.dummy import DummyClassifier
from classifier.classifiers.llm_classifier import LLMClassifier
from classifier.models import ClassificationResult, DocumentMetadata


def _fake_llm(*verdicts: str) -> FakeListChatModel:
    """Return a FakeListChatModel whose responses are JSON-encoded DoctorVisitOutput objects."""
    responses = [json.dumps({"verdict": v}) for v in verdicts]
    return FakeListChatModel(responses=responses)


def _invalid_llm() -> FakeListChatModel:
    """Return a FakeListChatModel that responds with unparseable text."""
    return FakeListChatModel(responses=["I'm not sure, maybe?"])


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

async def test_doctor_visit_needed_returns_needs_visit(
    sample_document_metadata: DocumentMetadata,
) -> None:
    clf = DoctorVisitNeededClassifier(llm=_fake_llm("NEEDS_VISIT"))
    result = await clf.run(sample_document_metadata)
    assert result == "NEEDS_VISIT"


async def test_doctor_visit_needed_returns_no_visit_needed(
    sample_document_metadata: DocumentMetadata,
) -> None:
    clf = DoctorVisitNeededClassifier(llm=_fake_llm("NO_VISIT_NEEDED"))
    result = await clf.run(sample_document_metadata)
    assert result == "NO_VISIT_NEEDED"


async def test_doctor_visit_needed_defaults_on_parse_error(
    sample_document_metadata: DocumentMetadata,
) -> None:
    clf = DoctorVisitNeededClassifier(llm=_invalid_llm())
    result = await clf.run(sample_document_metadata)
    assert result == "NEEDS_VISIT"


async def test_doctor_visit_needed_task_name() -> None:
    assert DoctorVisitNeededClassifier.task_name == "doctor_visit_needed"


async def test_doctor_visit_needed_prompt_input_includes_complaints(
    sample_document_metadata: DocumentMetadata,
) -> None:
    clf = DoctorVisitNeededClassifier(llm=_fake_llm("NEEDS_VISIT"))
    prompt_input = clf._build_prompt_input(sample_document_metadata)
    assert "Well Visit" in prompt_input["complaints"]
    assert "Runny nose" in prompt_input["complaints"]


async def test_doctor_visit_needed_prompt_input_includes_assessment(
    sample_document_metadata: DocumentMetadata,
) -> None:
    clf = DoctorVisitNeededClassifier(llm=_fake_llm("NEEDS_VISIT"))
    prompt_input = clf._build_prompt_input(sample_document_metadata)
    assert "Z00.129" in prompt_input["assessment"]


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

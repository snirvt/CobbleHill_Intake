"""Integration test: the extraction step finds the right diagnoses in a synthetic note.

Exercises only extraction (no bucketing), so its prompt can be tuned separately from
categorization. Hits the configured model, so it needs that service running.

Run with: pytest -m integration tests/integration/test_diagnosis_extraction.py
"""

from pathlib import Path

import pytest

from classifier.classifiers.diagnosis_extraction import DiagnosisExtractionClassifier
from classifier.main import _task_llm
from classifier.models import DocumentMetadata, PatientMetadata


def _note(text: str) -> DocumentMetadata:
    return DocumentMetadata(file_path=Path("dr_note.txt"), raw_text=text, meta=PatientMetadata())


async def _names(text: str) -> tuple[list[str], list[str | None]]:
    result = await DiagnosisExtractionClassifier(llm=_task_llm("diagnosis")).extract_diagnoses(
        _note(text)
    )
    assert result.error is None, result.error
    return [d.name.lower() for d in result.diagnoses], [d.icd_code for d in result.diagnoses]


@pytest.mark.integration
async def test_assessment_diagnosis_wins_over_chief_complaint() -> None:
    names, _ = await _names(
        "Chief complaint: cough and fever\n\nAssessment:\n1. Community-acquired pneumonia\n"
    )

    assert any("pneumonia" in n for n in names)
    assert not any(n in {"cough", "fever"} for n in names)


@pytest.mark.integration
async def test_icd_codes_kept_when_present() -> None:
    names, codes = await _names(
        "Assessment:\n"
        "1. Underweight - R63.6\n"
        "2. Neonatal agitation - R68.12\n"
        "3. Vomiting in infants - R11.10\n"
    )

    assert len(names) == 3
    assert set(codes) == {"R63.6", "R68.12", "R11.10"}


@pytest.mark.integration
async def test_falls_back_to_hpi_without_assessment() -> None:
    names, codes = await _names("HPI: 3 week old with persistent colic and poor feeding.\n")

    assert any("colic" in n for n in names)
    assert all(code is None for code in codes)


@pytest.mark.integration
async def test_negative_findings_and_status_remarks_are_skipped() -> None:
    """Synthetic post-op note: real problems kept; "no ..." findings and "healing well" dropped."""
    names, _ = await _names(
        "Assessment:\n"
        "1. Post-operative day 10 status post L3-S1 lumbar laminectomy and decompression\n"
        "2. Post-operative pain improving on current regimen\n"
        "3. Hypertension\n"
        "4. Hyperlipidemia\n"
        "5. No respiratory symptoms at this time\n"
        "6. No evidence of latent tuberculosis infection\n"
        "7. Incision healing well\n"
    )

    assert any("laminectomy" in n for n in names), names
    assert any("hypertension" in n for n in names), names
    assert any("pain" in n for n in names), names
    assert not any(n.startswith("no ") for n in names), names
    assert not any("healing well" in n for n in names), names

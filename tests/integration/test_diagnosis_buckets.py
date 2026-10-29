"""Integration test: synthetic diagnoses land in the expected DiagnosisCategory bucket.

Hits the configured ollama model, so it needs that service running. Notes are synthetic.

Run with: pytest -m integration
"""

from pathlib import Path

import pytest

from classifier.main import build_diagnosis_extractor
from classifier.models import DiagnosisCategory, DocumentMetadata, PatientMetadata

_CASES = [
    ("Chronic venous stasis of bilateral lower extremities", DiagnosisCategory.LOWER_EXTREMITY),
    ("Venous stasis ulcer of left lower leg", DiagnosisCategory.LOWER_EXTREMITY),
    ("Stage 3 pressure injury of sacrum", DiagnosisCategory.WOUND),
    ("Diabetic foot ulcer, right foot", DiagnosisCategory.WOUND),
    ("Contact dermatitis of hands", DiagnosisCategory.SKIN),
    ("Alzheimer's dementia, moderate", DiagnosisCategory.MENTAL_HEALTH),
    ("Postpartum depression", DiagnosisCategory.MENTAL_HEALTH),
    ("Postpartum hemorrhage", DiagnosisCategory.WOMENS_HEALTH),
    ("Uterine pain, 30 weeks pregnant", DiagnosisCategory.WOMENS_HEALTH),
    ("Pelvic pain, uterine fibroids", DiagnosisCategory.WOMENS_HEALTH),
    ("Feeding difficulty in newborn", DiagnosisCategory.FEEDING),
]


@pytest.mark.integration
@pytest.mark.parametrize(("diagnosis", "expected"), _CASES)
async def test_diagnosis_lands_in_expected_bucket(
    diagnosis: str, expected: DiagnosisCategory
) -> None:
    text = f"Patient: EXAMPLE, ADULT\nAge: 70 years\n\nAssessment:\n1. {diagnosis}\n"
    metadata = DocumentMetadata(
        file_path=Path("dr_note.txt"), raw_text=text, meta=PatientMetadata()
    )

    result = await build_diagnosis_extractor().extract_diagnoses(metadata)

    assert result.error is None
    assert expected in result.diagnosis_categories, result.diagnoses

"""Integration test: diagnosis strings land in the expected DiagnosisCategory bucket.

Exercises only the bucketing step (no note, no extraction), so the categorization prompt
can be tuned against this list alone. Hits the configured model, so it needs that
service running. Diagnoses are synthetic.

Run with: pytest -m integration tests/integration/test_diagnosis_buckets.py
"""

import pytest

from classifier.main import build_diagnosis_categorizer
from classifier.models import Diagnosis, DiagnosisCategory

_CASES = [
    (
        Diagnosis(name="Chronic venous stasis of bilateral lower extremities"),
        DiagnosisCategory.LOWER_EXTREMITY,
    ),
    (Diagnosis(name="Venous stasis ulcer of left lower leg"), DiagnosisCategory.LOWER_EXTREMITY),
    (Diagnosis(name="Stage 3 pressure injury of sacrum"), DiagnosisCategory.WOUND),
    (Diagnosis(name="Diabetic foot ulcer, right foot"), DiagnosisCategory.WOUND),
    (Diagnosis(name="Contact dermatitis of hands"), DiagnosisCategory.SKIN),
    (Diagnosis(name="Alzheimer's dementia, moderate"), DiagnosisCategory.MENTAL_HEALTH),
    (Diagnosis(name="Postpartum depression"), DiagnosisCategory.MENTAL_HEALTH),
    (Diagnosis(name="Postpartum hemorrhage"), DiagnosisCategory.WOMENS_HEALTH),
    (Diagnosis(name="Uterine pain, 30 weeks pregnant"), DiagnosisCategory.WOMENS_HEALTH),
    (Diagnosis(name="Pelvic pain, uterine fibroids"), DiagnosisCategory.WOMENS_HEALTH),
    (Diagnosis(name="Feeding difficulty in newborn"), DiagnosisCategory.FEEDING),
    (Diagnosis(name="Feeding disability", icd_code="R63.30"), DiagnosisCategory.FEEDING),
    (Diagnosis(name="Underweight", icd_code="R63.6"), DiagnosisCategory.FEEDING),
    (Diagnosis(name="Short stature"), DiagnosisCategory.FEEDING),
    (
        Diagnosis(name="Vomiting in infants AND/OR children", icd_code="R11.10"),
        DiagnosisCategory.REFLUX,
    ),
    (Diagnosis(name="Neonatal agitation", icd_code="R68.12"), DiagnosisCategory.FUSSINESS),
    (Diagnosis(name="Colic"), DiagnosisCategory.FUSSINESS),
]


@pytest.mark.integration
@pytest.mark.parametrize(("diagnosis", "expected"), _CASES, ids=lambda v: getattr(v, "name", v))
async def test_diagnosis_lands_in_expected_bucket(
    diagnosis: Diagnosis, expected: DiagnosisCategory
) -> None:
    result = await build_diagnosis_categorizer().categorize([diagnosis])

    assert result.error is None, result.error
    assert result.diagnoses[0].category is expected, result.reasoning


@pytest.mark.integration
async def test_whole_note_batch_keeps_order_and_count() -> None:
    """The screenshot patient's six diagnoses in one call: one bucket each, in order."""
    batch = [d for d, _ in _CASES[-6:]] + [Diagnosis(name="Abdominal bloating", icd_code="R14.0")]

    result = await build_diagnosis_categorizer().categorize(batch)

    assert result.error is None, result.error
    assert [d.name for d in result.diagnoses] == [d.name for d in batch]
    assert [d.category for d in result.diagnoses[:6]] == [e for _, e in _CASES[-6:]]

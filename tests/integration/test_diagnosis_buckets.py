"""Integration test: diagnosis strings land in the expected DiagnosisCategory bucket.

Exercises only the bucketing step (no note, no extraction), so the categorization prompt
can be tuned against this list alone. Each case carries the age text (e.g. "3 months")
the pipelines pass, since age decides child-vs-adult buckets. Hits the configured model, so
it needs that service running. Diagnoses are synthetic.

Run with: pytest -m integration tests/integration/test_diagnosis_buckets.py
"""

import pytest

from classifier.main import build_diagnosis_categorizer
from classifier.models import Diagnosis, DiagnosisCategory

_ADULT = "70 years"
_CHILD = "3 months"

_CASES = [
    (
        Diagnosis(name="Chronic venous stasis of bilateral lower extremities"),
        _ADULT,
        DiagnosisCategory.LOWER_EXTREMITY,
    ),
    (
        Diagnosis(name="Venous stasis ulcer of left lower leg"),
        _ADULT,
        DiagnosisCategory.LOWER_EXTREMITY,
    ),
    (Diagnosis(name="Stage 3 pressure injury of sacrum"), _ADULT, DiagnosisCategory.WOUND),
    (Diagnosis(name="Diabetic foot ulcer, right foot"), _ADULT, DiagnosisCategory.WOUND),
    (Diagnosis(name="Contact dermatitis of hands"), _ADULT, DiagnosisCategory.SKIN),
    (Diagnosis(name="Alzheimer's dementia, moderate"), _ADULT, DiagnosisCategory.MENTAL_HEALTH),
    (Diagnosis(name="Postpartum depression"), "30 years", DiagnosisCategory.MENTAL_HEALTH),
    (Diagnosis(name="Postpartum hemorrhage"), "30 years", DiagnosisCategory.WOMENS_HEALTH),
    (
        Diagnosis(name="Uterine pain, 30 weeks pregnant"),
        "30 years",
        DiagnosisCategory.WOMENS_HEALTH,
    ),
    (Diagnosis(name="Pelvic pain, uterine fibroids"), "45 years", DiagnosisCategory.WOMENS_HEALTH),
    (Diagnosis(name="Underweight", icd_code="R63.6"), _ADULT, DiagnosisCategory.METABOLIC),
    (Diagnosis(name="Obesity, BMI 35"), _ADULT, DiagnosisCategory.METABOLIC),
    (Diagnosis(name="Feeding difficulty in newborn"), _CHILD, DiagnosisCategory.FEEDING),
    # The screenshot patient (0 days old); also the batch test below.
    (Diagnosis(name="Feeding disability", icd_code="R63.30"), _CHILD, DiagnosisCategory.FEEDING),
    (Diagnosis(name="Underweight", icd_code="R63.6"), _CHILD, DiagnosisCategory.FEEDING),
    (Diagnosis(name="Short stature"), _CHILD, DiagnosisCategory.FEEDING),
    (
        Diagnosis(name="Vomiting in infants AND/OR children", icd_code="R11.10"),
        _CHILD,
        DiagnosisCategory.REFLUX,
    ),
    (
        Diagnosis(name="Neonatal agitation", icd_code="R68.12"),
        _CHILD,
        DiagnosisCategory.FUSSINESS,
    ),
    (Diagnosis(name="Colic"), _CHILD, DiagnosisCategory.FUSSINESS),
]


def _case_id(value: object) -> str:
    return value.name if isinstance(value, Diagnosis) else str(value)


@pytest.mark.integration
@pytest.mark.parametrize(("diagnosis", "age", "expected"), _CASES, ids=_case_id)
async def test_diagnosis_lands_in_expected_bucket(
    diagnosis: Diagnosis, age: str, expected: DiagnosisCategory
) -> None:
    result = await build_diagnosis_categorizer().categorize([diagnosis], age)

    assert result.error is None, result.error
    assert result.diagnoses[0].category is expected, (result.diagnoses, result.reasoning)


@pytest.mark.integration
async def test_whole_note_batch_keeps_order_and_count() -> None:
    """The screenshot patient's six diagnoses in one call: one bucket each, in order."""
    peds = _CASES[-6:]
    batch = [d for d, _, _ in peds] + [Diagnosis(name="Abdominal bloating", icd_code="R14.0")]

    result = await build_diagnosis_categorizer().categorize(batch, "0 days")

    assert result.error is None, result.error
    assert [d.name for d in result.diagnoses] == [d.name for d in batch]
    assert [d.category for d in result.diagnoses[:6]] == [e for _, _, e in peds], (
        result.diagnoses,
        result.reasoning,
    )

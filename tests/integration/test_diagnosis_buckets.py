"""Integration test: diagnosis strings land in the expected DiagnosisCategory bucket.

Exercises only the bucketing step (no note, no extraction), so the categorization prompt
can be tuned against this list alone. Each case carries the age text the pipelines pass
(``describe_age``, e.g. "3 months (infant)"), since the infant label decides some buckets.
Hits the configured model, so it needs that service running. Diagnoses are synthetic.

Run with: pytest -m integration tests/integration/test_diagnosis_buckets.py
"""

import pytest

from classifier.main import build_diagnosis_categorizer
from classifier.models import Diagnosis, DiagnosisCategory

_ADULT = "70 years (not an infant)"
_TEEN = "16 years (not an infant)"
_INFANT = "3 months (infant)"

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
    (
        Diagnosis(name="Postpartum depression"),
        "30 years (not an infant)",
        DiagnosisCategory.MENTAL_HEALTH,
    ),
    (
        Diagnosis(name="Postpartum hemorrhage"),
        "30 years (not an infant)",
        DiagnosisCategory.WOMENS_HEALTH,
    ),
    (
        Diagnosis(name="Uterine pain, 30 weeks pregnant"),
        "30 years (not an infant)",
        DiagnosisCategory.WOMENS_HEALTH,
    ),
    (
        Diagnosis(name="Pelvic pain, uterine fibroids"),
        "45 years (not an infant)",
        DiagnosisCategory.WOMENS_HEALTH,
    ),
    (Diagnosis(name="Underweight", icd_code="R63.6"), _ADULT, DiagnosisCategory.METABOLIC),
    (Diagnosis(name="Underweight", icd_code="R63.6"), _TEEN, DiagnosisCategory.METABOLIC),
    (Diagnosis(name="Obesity, BMI 35"), _ADULT, DiagnosisCategory.METABOLIC),
    (Diagnosis(name="Feeding difficulty in newborn"), _INFANT, DiagnosisCategory.FEEDING),
    # Reviewer: short stature kept landing in Gastrointestinal; growth is Feeding at any age.
    (
        Diagnosis(name="Short stature", icd_code="R62.52"),
        "5 years (not an infant)",
        DiagnosisCategory.FEEDING,
    ),
    (Diagnosis(name="Short stature", icd_code="R62.52"), _TEEN, DiagnosisCategory.FEEDING),
    (Diagnosis(name="Short stature"), "8 years (not an infant)", DiagnosisCategory.FEEDING),
    (
        Diagnosis(name="Short stature", icd_code="R62.52"),
        "0 days (infant)",
        DiagnosisCategory.FEEDING,
    ),
    (
        Diagnosis(name="Lack of expected normal physiological development", icd_code="R62.50"),
        "4 years (not an infant)",
        DiagnosisCategory.FEEDING,
    ),
    # One merged digestive bucket: reflux, vomiting and bowel problems.
    (
        Diagnosis(name="Gastroesophageal reflux disease", icd_code="K21.9"),
        _INFANT,
        DiagnosisCategory.GASTROINTESTINAL,
    ),
    (Diagnosis(name="Constipation"), _ADULT, DiagnosisCategory.GASTROINTESTINAL),
    # A 75-year-old's negative finding landed in Newborn / Well Child.
    (
        Diagnosis(name="No respiratory symptoms at this time"),
        "75 years (not an infant)",
        DiagnosisCategory.OTHER,
    ),
    (
        Diagnosis(name="Routine well-child visit"),
        "4 years (not an infant)",
        DiagnosisCategory.NEWBORN,
    ),
    (Diagnosis(name="Routine newborn care"), "2 days (infant)", DiagnosisCategory.NEWBORN),
    # Underweight infant kept landing in Obesity / Metabolic.
    (
        Diagnosis(name="Underweight", icd_code="R63.6"),
        "0 days (infant)",
        DiagnosisCategory.FEEDING,
    ),
    (Diagnosis(name="Underweight"), "1 year (infant)", DiagnosisCategory.FEEDING),
    (Diagnosis(name="Poor weight gain"), "6 months (infant)", DiagnosisCategory.FEEDING),
    (Diagnosis(name="Underweight", icd_code="R63.6"), None, DiagnosisCategory.FEEDING),
    # The screenshot patient (0 days old); also the batch test below.
    (Diagnosis(name="Feeding disability", icd_code="R63.30"), _INFANT, DiagnosisCategory.FEEDING),
    (Diagnosis(name="Underweight", icd_code="R63.6"), _INFANT, DiagnosisCategory.FEEDING),
    (Diagnosis(name="Short stature", icd_code="R62.52"), _INFANT, DiagnosisCategory.FEEDING),
    (
        Diagnosis(name="Vomiting in infants AND/OR children", icd_code="R11.10"),
        _INFANT,
        DiagnosisCategory.GASTROINTESTINAL,
    ),
    (
        Diagnosis(name="Neonatal agitation", icd_code="R68.12"),
        _INFANT,
        DiagnosisCategory.FUSSINESS,
    ),
    (Diagnosis(name="Colic"), _INFANT, DiagnosisCategory.FUSSINESS),
]


def _why(diagnoses: list[Diagnosis]) -> list[str]:
    """Each diagnosis with the bucket the model picked and its reason, for failure messages."""
    return [f"{d.name} -> {d.category}: {d.category_reason}" for d in diagnoses]


def _case_id(value: object) -> str:
    if isinstance(value, Diagnosis):
        return f"{value.name} ({value.icd_code})" if value.icd_code else value.name
    return "unknown age" if value is None else str(value)


@pytest.mark.integration
@pytest.mark.parametrize(("diagnosis", "age", "expected"), _CASES, ids=_case_id)
async def test_diagnosis_lands_in_expected_bucket(
    diagnosis: Diagnosis, age: str | None, expected: DiagnosisCategory
) -> None:
    result = await build_diagnosis_categorizer().categorize([diagnosis], age)

    assert result.error is None, result.error
    assert result.diagnoses[0].category is expected, _why(result.diagnoses)


@pytest.mark.integration
async def test_adult_never_lands_in_newborn_well_child() -> None:
    diagnosis = Diagnosis(name="Encounter for general adult medical examination")

    result = await build_diagnosis_categorizer().categorize([diagnosis], _ADULT)

    assert result.error is None, result.error
    assert result.diagnoses[0].category is not DiagnosisCategory.NEWBORN, _why(result.diagnoses)


@pytest.mark.integration
async def test_whole_note_batch_keeps_order_and_count() -> None:
    """The screenshot patient's six diagnoses in one call: one bucket each, in order."""
    peds = _CASES[-6:]
    batch = [d for d, _, _ in peds] + [Diagnosis(name="Abdominal bloating", icd_code="R14.0")]

    result = await build_diagnosis_categorizer().categorize(batch, "0 days (infant)")

    assert result.error is None, result.error
    assert [d.name for d in result.diagnoses] == [d.name for d in batch]
    expected = [e for _, _, e in peds] + [DiagnosisCategory.GASTROINTESTINAL]
    assert [d.category for d in result.diagnoses] == expected, _why(result.diagnoses)

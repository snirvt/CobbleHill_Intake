import re

import pytest

from classifier.classifiers.categories import (
    CATEGORY_EXAMPLES,
    CATEGORY_LIST,
    CATEGORY_RULES,
    describe_age,
)
from classifier.models import Diagnosis, DiagnosisCategory, unique_categories
from config.settings import settings


def test_category_list_names_every_category() -> None:
    for category in DiagnosisCategory:
        assert f"- {category.value}" in CATEGORY_LIST


def test_category_rules_reference_only_known_categories() -> None:
    known = {category.value for category in DiagnosisCategory}

    quoted = set(re.findall(r'"([^"]+)"', CATEGORY_RULES))

    assert quoted
    assert quoted <= known


def test_category_rules_route_venous_wound_and_dementia() -> None:
    assert '"Lower Extremity / Venous"' in CATEGORY_RULES
    assert '"Wound"' in CATEGORY_RULES
    assert "Dementia" in CATEGORY_RULES
    assert "Postpartum depression" in CATEGORY_RULES


def _rule(keyword: str) -> str:
    """The whole rule bullet (joined across wrapped lines) that mentions keyword."""
    return next(rule for rule in CATEGORY_RULES.split("\n- ") if keyword in rule)


def test_category_rules_route_uterine_pain_to_womens_health() -> None:
    womens_health_rule = _rule("Postpartum depression")
    reflux_rule = _rule("heartburn")

    assert "uterine or pelvic pain" in womens_health_rule
    assert '"Women\'s Health"' in womens_health_rule
    assert "never use" in reflux_rule and "uterine" in reflux_rule


def test_category_rules_route_newborn_feeding_to_feeding_growth() -> None:
    feeding_rule = _rule("failure to")

    assert '"Feeding / Growth"' in feeding_rule


def test_category_rules_newborn_well_child_never_for_adults() -> None:
    newborn_rule = _rule('"Newborn / Well Child" is only')

    assert "infant or child" in newborn_rule
    assert "never for an adult" in newborn_rule
    assert '(→ "Feeding / Growth")' in newborn_rule


def test_category_rules_send_negative_findings_to_other() -> None:
    negative_rule = _rule("negative finding")

    assert "no respiratory symptoms" in negative_rule
    assert 'is "Other"' in negative_rule


def test_category_rules_route_vomiting_to_reflux() -> None:
    vomiting_rule = _rule("spit-up")

    assert "vomiting in infants" in vomiting_rule
    assert '"Reflux / GERD / Vomiting"' in vomiting_rule
    assert '"Infection / Sepsis"' in vomiting_rule


def test_category_rules_route_growth_terms_to_feeding_growth() -> None:
    feeding_rule = _rule("failure to")

    for term in ("short stature", "feeding disability"):
        assert term in feeding_rule
    assert "underweight" not in feeding_rule
    assert '"Blood / Anemia"' in feeding_rule
    assert '"Developmental / Cognitive"' in feeding_rule


def test_category_rules_weight_problems_depend_on_infant_label() -> None:
    weight_rule = _rule("Weight problems")

    assert "underweight" in weight_rule and "R63.6" in weight_rule
    infant = weight_rule.index('are "Feeding / Growth" when the patient is an infant or the age is')
    adult = weight_rule.index('labelled not an infant are they "Obesity / Metabolic"')
    assert infant < adult  # the infant / unknown branch comes first
    assert "years" not in weight_rule  # the infant cutoff stays hidden from the model


def test_category_rules_growth_terms_apply_at_any_age_with_icd_hints() -> None:
    feeding_rule = _rule("failure to")

    assert '"Feeding / Growth" at any age' in feeding_rule
    assert "short stature (R62.52)" in feeding_rule
    assert '"Gastrointestinal"' in feeding_rule


def test_category_examples_cover_the_reported_misses() -> None:
    lines = CATEGORY_EXAMPLES.splitlines()

    assert "- Short stature (R62.52), 5 years (not an infant) → Feeding / Growth" in lines
    assert "- Underweight (R63.6), 0 days (infant) → Feeding / Growth" in lines
    assert "- Underweight (R63.6), 40 years (not an infant) → Obesity / Metabolic" in lines
    assert "- No respiratory symptoms at this time, 75 years (not an infant) → Other" in lines


def test_category_examples_end_in_known_categories() -> None:
    known = {category.value for category in DiagnosisCategory}

    targets = {line.rsplit(" → ", 1)[1] for line in CATEGORY_EXAMPLES.splitlines()}

    assert targets <= known


@pytest.mark.parametrize(
    ("age_years", "expected"),
    [
        (3 / 12, "3 months (infant)"),
        (0.0, "0 days (infant)"),
        (2.9, "2 years (infant)"),
        (3.0, "3 years (not an infant)"),
        (16.0, "16 years (not an infant)"),
        (None, None),
    ],
)
def test_describe_age_labels_infants(age_years: float | None, expected: str | None) -> None:
    assert describe_age(age_years) == expected


def test_describe_age_uses_configurable_infant_cutoff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "infant_age_threshold_years", 1)

    assert describe_age(1.5) == "18 months (not an infant)"


def test_category_rules_route_agitation_and_colic_to_fussiness() -> None:
    fussiness_rule = _rule("neonatal agitation")

    assert "colic" in fussiness_rule
    assert '"Fussiness / Irritability"' in fussiness_rule
    assert '"Brain Bleed / Stroke"' in fussiness_rule


def test_unique_categories_dedupes_in_first_appearance_order() -> None:
    diagnoses = [
        Diagnosis(name="Pneumonia", category=DiagnosisCategory.RESPIRATORY),
        Diagnosis(name="Hypertension", category=DiagnosisCategory.CARDIOVASCULAR),
        Diagnosis(name="Asthma", category=DiagnosisCategory.RESPIRATORY),
    ]

    assert unique_categories(diagnoses) == [
        DiagnosisCategory.RESPIRATORY,
        DiagnosisCategory.CARDIOVASCULAR,
    ]


def test_unique_categories_skips_missing_category() -> None:
    diagnoses = [Diagnosis(name="X"), Diagnosis(name="GERD", category=DiagnosisCategory.REFLUX)]

    assert unique_categories(diagnoses) == [DiagnosisCategory.REFLUX]


def test_unique_categories_empty_for_no_diagnoses() -> None:
    assert unique_categories([]) == []

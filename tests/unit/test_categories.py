import re

from classifier.classifiers.categories import CATEGORY_LIST, CATEGORY_RULES
from classifier.models import Diagnosis, DiagnosisCategory, unique_categories


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
    assert '"Newborn / Well Child"' in feeding_rule


def test_category_rules_route_vomiting_to_reflux() -> None:
    vomiting_rule = _rule("spit-up")

    assert "vomiting in infants" in vomiting_rule
    assert '"Reflux / GERD / Vomiting"' in vomiting_rule
    assert '"Infection / Sepsis"' in vomiting_rule


def test_category_rules_route_growth_terms_to_feeding_growth() -> None:
    feeding_rule = _rule("failure to")

    for term in ("short stature", "underweight", "feeding disability", "child"):
        assert term in feeding_rule
    assert '"Obesity / Metabolic"' in feeding_rule
    assert '"Blood / Anemia"' in feeding_rule
    assert '"Developmental / Cognitive"' in feeding_rule


def test_category_rules_route_adult_underweight_to_metabolic() -> None:
    adult_rule = _rule("In an adult")

    assert "underweight" in adult_rule
    assert '"Obesity / Metabolic"' in adult_rule


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

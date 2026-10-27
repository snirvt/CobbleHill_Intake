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

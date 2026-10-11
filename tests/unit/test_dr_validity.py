from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from classifier.classifiers.dr_validity import DrNoteValidityClassifier
from classifier.models import (
    CareCheck,
    Diagnosis,
    DiagnosisCheck,
    DiagnosisExtractionResult,
    DocumentMetadata,
    MainDiagnosisResult,
    PatientAgeResult,
    PatientGroup,
    PatientMetadata,
    TreatmentRequestResult,
    ValidityVerdict,
)

_DR_PATHS = [Path("Pending/DOE, JANE/dr_a.pdf"), Path("Pending/DOE, JANE/dr_b.pdf")]


def _make_classifier(
    age: float | None,
    has_diagnosis: bool = True,
    necessity: bool = True,
    main: str | None = "Colic",
) -> tuple[DrNoteValidityClassifier, dict[str, AsyncMock]]:
    mocks = {
        "age": AsyncMock(
            side_effect=lambda m: PatientAgeResult(file_path=m.file_path, age_years=age)
        ),
        "diagnosis": AsyncMock(
            side_effect=lambda m: DiagnosisExtractionResult(
                file_path=m.file_path,
                diagnoses=[Diagnosis(name="Colic", icd_code="R10.83")] if has_diagnosis else [],
            )
        ),
        "care": AsyncMock(
            side_effect=lambda m: TreatmentRequestResult(
                file_path=m.file_path, treatment_requested=necessity, reasoning="needs home care"
            )
        ),
        "main": AsyncMock(
            side_effect=lambda m: MainDiagnosisResult(file_path=m.file_path, main_diagnosis=main)
        ),
    }
    age_x, dx_x, care_x, main_x = MagicMock(), MagicMock(), MagicMock(), MagicMock()
    age_x.extract_age = mocks["age"]
    dx_x.extract_diagnoses = mocks["diagnosis"]
    care_x.extract_treatment_request = mocks["care"]
    main_x.extract_main_diagnosis = mocks["main"]
    clf = DrNoteValidityClassifier(
        age_extractor=age_x,
        diagnosis_extractor=dx_x,
        care_extractor=care_x,
        main_diagnosis_extractor=main_x,
    )
    return clf, mocks


def _metadata(text: str = "--- dr_a.pdf ---\nA\n\n--- dr_b.pdf ---\nB") -> DocumentMetadata:
    return DocumentMetadata(file_path=_DR_PATHS[0], raw_text=text, meta=PatientMetadata())


# ---------------------------------------------------------------------------
# _patient_group
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("age", "group"),
    [
        (0.5, PatientGroup.PEDIATRIC),
        (17.99, PatientGroup.PEDIATRIC),
        (18.0, PatientGroup.ADULT),
        (85, PatientGroup.ADULT),
        (None, PatientGroup.UNKNOWN),
    ],
)
def test_patient_group_uses_threshold(age: float | None, group: PatientGroup) -> None:
    assert DrNoteValidityClassifier._patient_group(age) is group


def test_patient_group_threshold_is_configurable(monkeypatch: pytest.MonkeyPatch) -> None:
    from config.settings import settings

    monkeypatch.setattr(settings, "pediatric_age_threshold", 21)
    assert DrNoteValidityClassifier._patient_group(19) is PatientGroup.PEDIATRIC


# ---------------------------------------------------------------------------
# _verdict
# ---------------------------------------------------------------------------

_EXISTS, _MISSING = DiagnosisCheck.EXISTS, DiagnosisCheck.MISSING
_NEEDED, _NOT_NEEDED = CareCheck.NEEDED, CareCheck.NOT_NEEDED
_PEDS, _ADULT, _UNKNOWN = PatientGroup.PEDIATRIC, PatientGroup.ADULT, PatientGroup.UNKNOWN


@pytest.mark.parametrize(
    ("group", "diagnosis", "care", "verdict", "reasons"),
    [
        (_PEDS, _EXISTS, _NEEDED, ValidityVerdict.VALID, []),
        (_PEDS, _EXISTS, _NOT_NEEDED, ValidityVerdict.INVALID, ["no medical necessity"]),
        (_PEDS, _MISSING, _NEEDED, ValidityVerdict.INVALID, ["no diagnosis"]),
        (
            _PEDS, _MISSING, _NOT_NEEDED, ValidityVerdict.INVALID,
            ["no diagnosis", "no medical necessity"],
        ),
        (_ADULT, _EXISTS, _NOT_NEEDED, ValidityVerdict.VALID, []),
        (_ADULT, _MISSING, _NEEDED, ValidityVerdict.INVALID, ["no diagnosis"]),
        (_UNKNOWN, _EXISTS, _NEEDED, ValidityVerdict.UNKNOWN_AGE, ["age not found"]),
        (
            _UNKNOWN, _EXISTS, _NOT_NEEDED, ValidityVerdict.UNKNOWN_AGE,
            ["age not found", "no medical necessity (required if under 18)"],
        ),
        (_UNKNOWN, _MISSING, _NEEDED, ValidityVerdict.INVALID, ["no diagnosis", "age not found"]),
    ],
)
def test_verdict_rules(
    group: PatientGroup,
    diagnosis: DiagnosisCheck,
    care: CareCheck,
    verdict: ValidityVerdict,
    reasons: list[str],
) -> None:
    assert DrNoteValidityClassifier._verdict(group, diagnosis, care) == (verdict, reasons)


# ---------------------------------------------------------------------------
# classify
# ---------------------------------------------------------------------------

async def test_classify_peds_valid_populates_all_fields() -> None:
    clf, _ = _make_classifier(age=0.67)

    result = await clf.classify(_metadata(), _DR_PATHS)

    assert result.patient_folder == Path("Pending/DOE, JANE")
    assert result.dr_paths == _DR_PATHS
    assert result.age_years == 0.67
    assert result.patient_group is PatientGroup.PEDIATRIC
    assert result.verdict is ValidityVerdict.VALID
    assert result.diagnosis_check is DiagnosisCheck.EXISTS
    assert [d.name for d in result.diagnoses] == ["Colic"]
    assert result.care_check is CareCheck.NEEDED
    assert result.care_reasoning == "needs home care"
    assert result.main_diagnosis == "Colic"


async def test_classify_peds_without_necessity_is_invalid() -> None:
    clf, _ = _make_classifier(age=5, necessity=False)

    result = await clf.classify(_metadata(), _DR_PATHS)

    assert result.verdict is ValidityVerdict.INVALID
    assert result.reasons == ["no medical necessity"]


async def test_classify_adult_ignores_necessity() -> None:
    clf, _ = _make_classifier(age=82, necessity=False, main="Dementia")

    result = await clf.classify(_metadata(), _DR_PATHS)

    assert result.verdict is ValidityVerdict.VALID
    assert result.care_check is None
    assert result.care_reasoning == ""
    assert result.main_diagnosis == "Dementia"


async def test_classify_unknown_age() -> None:
    clf, _ = _make_classifier(age=None)

    result = await clf.classify(_metadata(), _DR_PATHS)

    assert result.verdict is ValidityVerdict.UNKNOWN_AGE
    assert result.patient_group is PatientGroup.UNKNOWN
    assert result.care_check is CareCheck.NEEDED  # kept so a reviewer can judge


async def test_classify_calls_every_extractor_once_with_combined_text() -> None:
    clf, mocks = _make_classifier(age=10)
    metadata = _metadata("--- dr_a.pdf ---\nDiagnosis\n\n--- dr_b.pdf ---\nNecessity")

    await clf.classify(metadata, _DR_PATHS)

    for mock in mocks.values():
        mock.assert_awaited_once()
        assert mock.await_args.args[0].raw_text == metadata.raw_text

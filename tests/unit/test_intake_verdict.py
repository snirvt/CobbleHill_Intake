from pathlib import Path

import pytest

from classifier.classifiers.intake_verdict import (
    REASON_NO_DIAGNOSIS,
    REASON_NO_NECESSITY,
    WARNING_UNKNOWN_AGE,
    explain,
    final_verdict,
    format_diagnosis_names,
    phase_1_verdict,
    phase_2_verdict,
)
from classifier.models import (
    CareCheck,
    ClinicalVerdict,
    Diagnosis,
    DiagnosisCheck,
    DrValidityResult,
    IntakeVerdict,
    PairMatch,
    PatientGroup,
    PhaseVerdict,
    ValidityVerdict,
)

_GERD = Diagnosis(name="GERD", icd_code="K21.9")
_ASTHMA = Diagnosis(name="Asthma")


def _phase_1(
    group: PatientGroup, diagnosis: bool = True, care: CareCheck | None = None
) -> DrValidityResult:
    return DrValidityResult(
        patient_folder=Path("/p"),
        patient_group=group,
        diagnosis_check=DiagnosisCheck.EXISTS if diagnosis else DiagnosisCheck.MISSING,
        diagnoses=[_GERD] if diagnosis else [],
        care_check=care,
        verdict=ValidityVerdict.VALID,
    )


def _pair(verdict: ClinicalVerdict, nurse: list[Diagnosis], category: str | None = None):
    return PairMatch(category=category, clinical_verdict=verdict, nurse_diagnoses=nurse)


# ---------------------------------------------------------------------------
# phase_1_verdict
# ---------------------------------------------------------------------------

def test_adult_with_diagnosis_passes_phase_1_without_necessity() -> None:
    assert phase_1_verdict(_phase_1(PatientGroup.ADULT)) == (PhaseVerdict.PASS, [], [])


def test_adult_without_diagnosis_fails_phase_1() -> None:
    verdict, reasons, _ = phase_1_verdict(_phase_1(PatientGroup.ADULT, diagnosis=False))
    assert verdict is PhaseVerdict.FAIL
    assert reasons == [REASON_NO_DIAGNOSIS]


def test_pediatric_needs_diagnosis_and_necessity() -> None:
    ok = _phase_1(PatientGroup.PEDIATRIC, care=CareCheck.NEEDED)
    no_need = _phase_1(PatientGroup.PEDIATRIC, care=CareCheck.NOT_NEEDED)
    neither = _phase_1(PatientGroup.PEDIATRIC, diagnosis=False, care=CareCheck.NOT_NEEDED)

    assert phase_1_verdict(ok)[0] is PhaseVerdict.PASS
    assert phase_1_verdict(no_need)[:2] == (PhaseVerdict.FAIL, [REASON_NO_NECESSITY])
    assert phase_1_verdict(neither)[1] == [REASON_NO_DIAGNOSIS, REASON_NO_NECESSITY]


@pytest.mark.parametrize(
    ("care", "expected"),
    [(CareCheck.NEEDED, PhaseVerdict.PASS), (CareCheck.NOT_NEEDED, PhaseVerdict.FAIL)],
)
def test_unknown_age_applies_pediatric_rules_with_warning(
    care: CareCheck, expected: PhaseVerdict
) -> None:
    verdict, _, warnings = phase_1_verdict(_phase_1(PatientGroup.UNKNOWN, care=care))
    assert verdict is expected
    assert warnings == [WARNING_UNKNOWN_AGE]


# ---------------------------------------------------------------------------
# phase_2_verdict / final_verdict
# ---------------------------------------------------------------------------

def test_phase_2_not_run_without_pairs() -> None:
    assert phase_2_verdict([]) is PhaseVerdict.NOT_RUN


def test_phase_2_passes_only_when_every_pair_matches() -> None:
    match = _pair(ClinicalVerdict.MATCH, [])
    assert phase_2_verdict([match, match]) is PhaseVerdict.PASS
    assert phase_2_verdict([match, _pair(ClinicalVerdict.MISMATCH, [])]) is PhaseVerdict.FAIL
    assert phase_2_verdict([_pair(ClinicalVerdict.UNDECIDED, [])]) is PhaseVerdict.FAIL


@pytest.mark.parametrize(
    ("p1", "p2", "expected"),
    [
        (PhaseVerdict.FAIL, PhaseVerdict.NOT_RUN, IntakeVerdict.REJECT),
        (PhaseVerdict.PASS, PhaseVerdict.PASS, IntakeVerdict.ACCEPT),
        (PhaseVerdict.PASS, PhaseVerdict.FAIL, IntakeVerdict.REJECT),
        (PhaseVerdict.PASS, PhaseVerdict.NOT_RUN, IntakeVerdict.INCOMPLETE),
    ],
)
def test_final_verdict_combines_phases(
    p1: PhaseVerdict, p2: PhaseVerdict, expected: IntakeVerdict
) -> None:
    assert final_verdict(p1, p2) is expected


# ---------------------------------------------------------------------------
# explain
# ---------------------------------------------------------------------------

def test_explain_phase_1_failure_lists_every_reason() -> None:
    text = explain(PhaseVerdict.FAIL, [REASON_NO_DIAGNOSIS, REASON_NO_NECESSITY], [], [])
    assert text == (
        "Patient did not pass Phase I: no valid diagnosis and no statement of medical necessity."
    )


def test_explain_match_quotes_both_notes() -> None:
    text = explain(PhaseVerdict.PASS, [], [_GERD], [_pair(ClinicalVerdict.MATCH, [_GERD])])
    assert text == (
        "Doctor's note says GERD (K21.9) and nurse confirmed and reported GERD (K21.9). "
        "The two reports match."
    )


def test_explain_mismatch_quotes_both_notes() -> None:
    text = explain(PhaseVerdict.PASS, [], [_GERD], [_pair(ClinicalVerdict.MISMATCH, [_ASTHMA])])
    assert text == (
        "Patient passed Phase I, but doctor said GERD (K21.9) and nurse said Asthma. "
        "The two reports do NOT match."
    )


def test_explain_nurse_without_diagnosis() -> None:
    text = explain(PhaseVerdict.PASS, [], [_GERD], [_pair(ClinicalVerdict.MISMATCH, [])])
    assert "nurse said no diagnosis" in text


def test_explain_without_pair_says_phase_2_not_run() -> None:
    assert "Phase II not run" in explain(PhaseVerdict.PASS, [], [_GERD], [])


def test_explain_multiple_pairs_prefixes_each_category() -> None:
    pairs = [
        _pair(ClinicalVerdict.MATCH, [_GERD], "hospital"),
        _pair(ClinicalVerdict.MISMATCH, [_ASTHMA], "peds"),
    ]
    text = explain(PhaseVerdict.PASS, [], [_GERD], pairs)
    assert text.startswith("[hospital] Doctor's note says")
    assert (
        "[peds] Patient passed Phase I, but doctor said GERD (K21.9) and nurse said Asthma"
        in text
    )


def test_format_diagnosis_names_adds_icd_when_present() -> None:
    assert format_diagnosis_names([_GERD, _ASTHMA]) == "GERD (K21.9); Asthma"

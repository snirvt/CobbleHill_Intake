from pathlib import Path

import pytest

from classifier.classifiers.intake_verdict import (
    REASON_NO_DIAGNOSIS,
    REASON_NO_NECESSITY,
    WARNING_UNKNOWN_AGE,
    dr_nurse_compare,
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


def test_phase_2_undecided_when_a_pair_is_undecided_and_none_mismatch() -> None:
    match = _pair(ClinicalVerdict.MATCH, [])
    undecided = _pair(ClinicalVerdict.UNDECIDED, [])
    mismatch = _pair(ClinicalVerdict.MISMATCH, [])
    assert phase_2_verdict([match, undecided]) is PhaseVerdict.UNDECIDED
    assert phase_2_verdict([undecided, mismatch]) is PhaseVerdict.FAIL


@pytest.mark.parametrize(
    ("p1", "p2", "expected"),
    [
        (PhaseVerdict.FAIL, PhaseVerdict.NOT_RUN, IntakeVerdict.REJECT),
        (PhaseVerdict.PASS, PhaseVerdict.PASS, IntakeVerdict.ACCEPT),
        (PhaseVerdict.PASS, PhaseVerdict.FAIL, IntakeVerdict.REJECT),
        (PhaseVerdict.PASS, PhaseVerdict.UNDECIDED, IntakeVerdict.NEEDS_REVIEW),
    ],
)
def test_final_verdict_combines_phases(
    p1: PhaseVerdict, p2: PhaseVerdict, expected: IntakeVerdict
) -> None:
    assert final_verdict(p1, p2, has_nurse_pair=True) is expected


@pytest.mark.parametrize("p1", [PhaseVerdict.PASS, PhaseVerdict.FAIL])
def test_final_verdict_no_nurse_note_whatever_phase_1_found(p1: PhaseVerdict) -> None:
    assert final_verdict(p1, None, has_nurse_pair=False) is IntakeVerdict.NO_NURSE_NOTE


# ---------------------------------------------------------------------------
# explain
# ---------------------------------------------------------------------------

def test_explain_phase_1_failure_lists_every_reason() -> None:
    text = explain(PhaseVerdict.FAIL, [REASON_NO_DIAGNOSIS, REASON_NO_NECESSITY], [], [], True)
    assert text == "Phase I failed: no valid diagnosis and no statement of medical necessity."


def test_explain_match_names_the_diagnoses() -> None:
    pairs = [_pair(ClinicalVerdict.MATCH, [_GERD])]
    text = explain(PhaseVerdict.PASS, [], [_GERD], pairs, True)
    assert text == "Doctor and nurse both report GERD (K21.9)."


def test_explain_mismatch_quotes_both_notes() -> None:
    pairs = [_pair(ClinicalVerdict.MISMATCH, [_ASTHMA])]
    text = explain(PhaseVerdict.PASS, [], [_GERD], pairs, True)
    assert text == "Notes do not match: doctor reports GERD (K21.9), nurse reports Asthma."


def test_explain_undecided_says_needs_review() -> None:
    pairs = [_pair(ClinicalVerdict.UNDECIDED, [_ASTHMA])]
    text = explain(PhaseVerdict.PASS, [], [_GERD], pairs, True)
    assert text == (
        "Match unclear, needs review: doctor reports GERD (K21.9), nurse reports Asthma."
    )


def test_explain_nurse_without_diagnosis() -> None:
    pairs = [_pair(ClinicalVerdict.MISMATCH, [])]
    text = explain(PhaseVerdict.PASS, [], [_GERD], pairs, True)
    assert text is not None and "nurse reports no diagnosis" in text


@pytest.mark.parametrize("p1", [PhaseVerdict.PASS, PhaseVerdict.FAIL])
def test_explain_is_none_without_nurse_note(p1: PhaseVerdict) -> None:
    assert explain(p1, [REASON_NO_DIAGNOSIS], [_GERD], [], False) is None


def test_explain_multiple_pairs_lists_only_non_matching_categories() -> None:
    pairs = [
        _pair(ClinicalVerdict.MATCH, [_GERD], "hospital"),
        _pair(ClinicalVerdict.MISMATCH, [_ASTHMA], "peds"),
    ]
    text = explain(PhaseVerdict.PASS, [], [_GERD], pairs, True)
    assert text == "[peds] Notes do not match: doctor reports GERD (K21.9), nurse reports Asthma."


def test_explain_multiple_matching_pairs_is_one_sentence() -> None:
    pairs = [
        _pair(ClinicalVerdict.MATCH, [_GERD], "hospital"),
        _pair(ClinicalVerdict.MATCH, [_GERD], "peds"),
    ]
    text = explain(PhaseVerdict.PASS, [], [_GERD], pairs, True)
    assert text == "Doctor and nurse both report GERD (K21.9)."


# ---------------------------------------------------------------------------
# dr_nurse_compare
# ---------------------------------------------------------------------------

def test_dr_nurse_compare_is_none_without_pairs() -> None:
    assert dr_nurse_compare([]) is None


def test_dr_nurse_compare_single_pair_is_its_reasoning() -> None:
    pair = PairMatch(clinical_verdict=ClinicalVerdict.MATCH, clinical_reasoning="Same story.")
    assert dr_nurse_compare([pair]) == "Same story."


def test_dr_nurse_compare_multiple_pairs_prefixes_each_category() -> None:
    pairs = [
        PairMatch(category="hospital", clinical_verdict=ClinicalVerdict.MATCH,
                  clinical_reasoning="A."),
        PairMatch(category="peds", clinical_verdict=ClinicalVerdict.MISMATCH,
                  clinical_reasoning="B."),
    ]
    assert dr_nurse_compare(pairs) == "[hospital] A. [peds] B."


def test_format_diagnosis_names_adds_icd_when_present() -> None:
    assert format_diagnosis_names([_GERD, _ASTHMA]) == "GERD (K21.9); Asthma"

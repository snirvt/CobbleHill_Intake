"""Intake-review verdict rules and explanation text: pure functions, no I/O.

Phase I (dr notes): adults need a diagnosis; pediatric patients — and patients whose age
could not be found — need a diagnosis AND a statement of medical necessity.
Phase II (only after a Phase I pass): every same-category dr + nurse pair must match;
an undecided match needs review. Without a same-category nurse note: NO_NURSE_NOTE.
"""

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
)
from classifier.naming import label

REASON_NO_DIAGNOSIS = "no valid diagnosis"
REASON_NO_NECESSITY = "no statement of medical necessity"
WARNING_UNKNOWN_AGE = "age not found — pediatric rules applied"
_NO_DIAGNOSIS = "no diagnosis"

_PHASE_1_FAILED = "Phase I failed: {reasons}."
_ALL_MATCH = "Doctor and nurse both report {dr}."
_PAIR_MISMATCH = "Notes do not match: doctor reports {dr}, nurse reports {nurse}."
_PAIR_UNDECIDED = "Match unclear, needs review: doctor reports {dr}, nurse reports {nurse}."


def phase_1_verdict(result: DrValidityResult) -> tuple[PhaseVerdict, list[str], list[str]]:
    """Apply the Phase I rules; return (verdict, what is missing, warnings).

    An unknown age gets the stricter pediatric rules plus a warning.
    """
    warnings = [WARNING_UNKNOWN_AGE] if result.patient_group is PatientGroup.UNKNOWN else []
    reasons: list[str] = []
    if result.diagnosis_check is DiagnosisCheck.MISSING:
        reasons.append(REASON_NO_DIAGNOSIS)
    if result.patient_group is not PatientGroup.ADULT and result.care_check is not CareCheck.NEEDED:
        reasons.append(REASON_NO_NECESSITY)
    verdict = PhaseVerdict.FAIL if reasons else PhaseVerdict.PASS
    return verdict, reasons, warnings


def phase_2_verdict(pairs: list[PairMatch]) -> PhaseVerdict:
    """NOT_RUN without pairs; FAIL on any mismatch, else UNDECIDED on any undecided pair."""
    if not pairs:
        return PhaseVerdict.NOT_RUN
    verdicts = {p.clinical_verdict for p in pairs}
    if ClinicalVerdict.MISMATCH in verdicts:
        return PhaseVerdict.FAIL
    if ClinicalVerdict.UNDECIDED in verdicts:
        return PhaseVerdict.UNDECIDED
    return PhaseVerdict.PASS


def final_verdict(
    phase_1: PhaseVerdict, phase_2: PhaseVerdict | None, has_nurse_pair: bool
) -> IntakeVerdict:
    """Combine both phases. No same-category nurse note wins over everything; then any
    failure rejects and an undecided match needs review."""
    if not has_nurse_pair:
        return IntakeVerdict.NO_NURSE_NOTE
    if phase_1 is PhaseVerdict.FAIL or phase_2 is PhaseVerdict.FAIL:
        return IntakeVerdict.REJECT
    if phase_2 is PhaseVerdict.UNDECIDED:
        return IntakeVerdict.NEEDS_REVIEW
    return IntakeVerdict.ACCEPT


def explain(
    phase_1: PhaseVerdict,
    phase_1_reasons: list[str],
    dr_diagnoses: list[Diagnosis],
    pairs: list[PairMatch],
    has_nurse_pair: bool,
) -> str | None:
    """Short reason for the final verdict, filled from the extracted data.

    None without a same-category nurse note. When every pair matches, one sentence;
    otherwise one sentence per non-matching pair, prefixed by its category when the
    patient has several pairs.
    """
    if not has_nurse_pair:
        return None
    if phase_1 is PhaseVerdict.FAIL:
        return _PHASE_1_FAILED.format(reasons=" and ".join(phase_1_reasons))
    dr = format_diagnosis_names(dr_diagnoses) or _NO_DIAGNOSIS
    off = [p for p in pairs if p.clinical_verdict is not ClinicalVerdict.MATCH]
    if not off:
        return _ALL_MATCH.format(dr=dr)
    sentences = []
    for pair in off:
        template = (
            _PAIR_MISMATCH if pair.clinical_verdict is ClinicalVerdict.MISMATCH else _PAIR_UNDECIDED
        )
        nurse = format_diagnosis_names(pair.nurse_diagnoses) or _NO_DIAGNOSIS
        sentence = template.format(dr=dr, nurse=nurse)
        sentences.append(f"[{label(pair.category)}] {sentence}" if len(pairs) > 1 else sentence)
    return " ".join(sentences)


def dr_nurse_compare(pairs: list[PairMatch]) -> str | None:
    """Phase II dr-vs-nurse comparison text: each pair's clinical reasoning, category-prefixed when
    there are several pairs; None without pairs."""
    if not pairs:
        return None
    if len(pairs) == 1:
        return pairs[0].clinical_reasoning
    return " ".join(f"[{label(p.category)}] {p.clinical_reasoning}" for p in pairs)


def format_diagnosis_names(diagnoses: list[Diagnosis]) -> str:
    """'name (ICD); name' — the diagnoses as they appear in an explanation."""
    return "; ".join(f"{d.name} ({d.icd_code})" if d.icd_code else d.name for d in diagnoses)

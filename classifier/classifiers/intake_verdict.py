"""Intake-review verdict rules and explanation text: pure functions, no I/O.

Phase I (dr notes): adults need a diagnosis; pediatric patients — and patients whose age
could not be found — need a diagnosis AND a statement of medical necessity.
Phase II (only after a Phase I pass): every same-category dr + nurse pair must match.
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

_PHASE_1_FAILED = "Patient did not pass Phase I: {reasons}."
_PAIR_MATCH = (
    "Doctor's note says {dr} and nurse confirmed and reported {nurse}. The two reports match."
)
_PAIR_MISMATCH = (
    "Patient passed Phase I, but doctor said {dr} and nurse said {nurse}. "
    "The two reports do NOT match."
)
_NO_PAIR = (
    "Patient passed Phase I, but has no nurse note of the same category; Phase II not run."
)


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
    """NOT_RUN without pairs; PASS only when every pair matches."""
    if not pairs:
        return PhaseVerdict.NOT_RUN
    if all(p.clinical_verdict is ClinicalVerdict.MATCH for p in pairs):
        return PhaseVerdict.PASS
    return PhaseVerdict.FAIL


def final_verdict(phase_1: PhaseVerdict, phase_2: PhaseVerdict) -> IntakeVerdict:
    """Combine both phases: any failure rejects; a skipped Phase II is incomplete."""
    if phase_1 is PhaseVerdict.FAIL or phase_2 is PhaseVerdict.FAIL:
        return IntakeVerdict.REJECT
    if phase_2 is PhaseVerdict.NOT_RUN:
        return IntakeVerdict.INCOMPLETE
    return IntakeVerdict.ACCEPT


def explain(
    phase_1: PhaseVerdict,
    phase_1_reasons: list[str],
    dr_diagnoses: list[Diagnosis],
    pairs: list[PairMatch],
) -> str:
    """Human-readable reason for the final verdict, filled from the extracted data.

    With several pairs, each gets its own sentence prefixed by its category.
    """
    if phase_1 is PhaseVerdict.FAIL:
        return _PHASE_1_FAILED.format(reasons=" and ".join(phase_1_reasons))
    if not pairs:
        return _NO_PAIR
    dr = format_diagnosis_names(dr_diagnoses) or _NO_DIAGNOSIS
    sentences = []
    for pair in pairs:
        template = (
            _PAIR_MATCH if pair.clinical_verdict is ClinicalVerdict.MATCH else _PAIR_MISMATCH
        )
        nurse = format_diagnosis_names(pair.nurse_diagnoses) or _NO_DIAGNOSIS
        sentence = template.format(dr=dr, nurse=nurse)
        sentences.append(f"[{label(pair.category)}] {sentence}" if len(pairs) > 1 else sentence)
    return " ".join(sentences)


def format_diagnosis_names(diagnoses: list[Diagnosis]) -> str:
    """'name (ICD); name' — the diagnoses as they appear in an explanation."""
    return "; ".join(f"{d.name} ({d.icd_code})" if d.icd_code else d.name for d in diagnoses)

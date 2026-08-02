import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from classifier.classifiers.dr_nurse_match import DrNurseMatchClassifier
from classifier.models import (
    Diagnosis,
    DiagnosisCheck,
    DiagnosisExtractionResult,
    DocumentMetadata,
    NursePatientMeta,
    NurseVisitFields,
    PairDocumentMetadata,
    PatientMetadata,
)


def _fake_llm(verdict: str, reasoning: str = "Ok.") -> FakeListChatModel:
    """Return a FakeListChatModel whose response is a JSON-encoded ClinicalMatchOutput."""
    return FakeListChatModel(responses=[json.dumps({"verdict": verdict, "reasoning": reasoning})])


def _invalid_llm() -> FakeListChatModel:
    return FakeListChatModel(responses=["not valid json"])


def _make_dr_meta(
    name: str = "Smith, John",
    dob: str = "01/15/2025",
    dos: str = "04/17/2026",
    sex: str = "M",
    account: str = "ABC123",
    raw_text: str = "Patient: Smith, John  DOB: 01/15/2025\nAssessment: Colic\nPlan: Follow up.",
) -> DocumentMetadata:
    return DocumentMetadata(
        file_path=Path("dr_note.pdf"),
        raw_text=raw_text,
        meta=PatientMetadata(
            patient_name=name,
            dob=dob,
            dos=dos,
            sex=sex,
            account_number=account,
        ),
    )


def _make_nurse_fields(
    name: str = "Smith, John",
    dob: str = "01/15/2025",
    dos: str = "04/17/2026",
    sex: str = "Male",
    prn: str = "ABC123",
) -> NurseVisitFields:
    return NurseVisitFields(
        meta=NursePatientMeta(patient_name=name, dob=dob, dos=dos, sex=sex, prn=prn),
    )


def _make_pair(
    dr: DocumentMetadata,
    nurse: NurseVisitFields,
    nurse_raw_text: str = "Date of service: 04/17/2026\nAssessment: Infantile colic - R10.83\nPlan: Monitor feeding.",
    category: str | None = None,
) -> PairDocumentMetadata:
    return PairDocumentMetadata(
        dr_file_path=Path("dr_note.pdf"),
        nurse_file_path=Path("nurse_note.pdf"),
        dr=dr,
        nurse=nurse,
        nurse_raw_text=nurse_raw_text,
        category=category,
    )


# ---------------------------------------------------------------------------
# Identity comparison
# ---------------------------------------------------------------------------

async def test_identity_all_match() -> None:
    clf = DrNurseMatchClassifier(llm=_fake_llm("MATCH"))
    pair = _make_pair(_make_dr_meta(), _make_nurse_fields())
    result = await clf.classify_pair(pair)
    assert result.identity_match.patient_name is True
    assert result.identity_match.dob is True
    assert result.identity_match.dos is True
    assert result.identity_match.sex is True
    assert result.identity_match.account_number is True


async def test_identity_name_mismatch() -> None:
    clf = DrNurseMatchClassifier(llm=_fake_llm("MATCH"))
    pair = _make_pair(
        _make_dr_meta(name="Smith, John"),
        _make_nurse_fields(name="Jones, Mary"),
    )
    result = await clf.classify_pair(pair)
    assert result.identity_match.patient_name is False


async def test_identity_dob_mismatch() -> None:
    clf = DrNurseMatchClassifier(llm=_fake_llm("MATCH"))
    pair = _make_pair(
        _make_dr_meta(dob="01/15/2025"),
        _make_nurse_fields(dob="06/20/2024"),
    )
    result = await clf.classify_pair(pair)
    assert result.identity_match.dob is False


async def test_identity_dos_two_digit_year_normalizes() -> None:
    clf = DrNurseMatchClassifier(llm=_fake_llm("MATCH"))
    pair = _make_pair(
        _make_dr_meta(dos="04/17/2026"),
        _make_nurse_fields(dos="04/17/26"),
    )
    result = await clf.classify_pair(pair)
    assert result.identity_match.dos is True


async def test_identity_sex_normalization_m_vs_male() -> None:
    clf = DrNurseMatchClassifier(llm=_fake_llm("MATCH"))
    pair = _make_pair(
        _make_dr_meta(sex="M"),
        _make_nurse_fields(sex="Male"),
    )
    result = await clf.classify_pair(pair)
    assert result.identity_match.sex is True


async def test_identity_sex_mismatch() -> None:
    clf = DrNurseMatchClassifier(llm=_fake_llm("MATCH"))
    pair = _make_pair(
        _make_dr_meta(sex="M"),
        _make_nurse_fields(sex="Female"),
    )
    result = await clf.classify_pair(pair)
    assert result.identity_match.sex is False


# ---------------------------------------------------------------------------
# Clinical verdict from LLM
# ---------------------------------------------------------------------------

async def test_clinical_verdict_match() -> None:
    clf = DrNurseMatchClassifier(
        llm=_fake_llm("MATCH", "Both notes describe the same infant.")
    )
    result = await clf.classify_pair(_make_pair(_make_dr_meta(), _make_nurse_fields()))
    assert result.clinical_verdict == "MATCH"
    assert "infant" in result.clinical_reasoning.lower()



async def test_clinical_verdict_mismatch() -> None:
    clf = DrNurseMatchClassifier(llm=_fake_llm("MISMATCH", "Different diagnoses entirely."))
    result = await clf.classify_pair(_make_pair(_make_dr_meta(), _make_nurse_fields()))
    assert result.clinical_verdict == "MISMATCH"


# ---------------------------------------------------------------------------
# Overall verdict — mirrors clinical_verdict only
# ---------------------------------------------------------------------------

async def test_overall_equals_clinical_verdict_match() -> None:
    clf = DrNurseMatchClassifier(llm=_fake_llm("MATCH", "Clinically consistent."))
    result = await clf.classify_pair(_make_pair(_make_dr_meta(), _make_nurse_fields()))
    assert result.overall == "MATCH"


async def test_overall_equals_clinical_verdict_mismatch() -> None:
    clf = DrNurseMatchClassifier(llm=_fake_llm("MISMATCH", "Completely different cases."))
    result = await clf.classify_pair(_make_pair(_make_dr_meta(), _make_nurse_fields()))
    assert result.overall == "MISMATCH"


async def test_overall_unaffected_by_identity_mismatch() -> None:
    """Identity mismatches no longer influence overall — clinical verdict wins."""
    clf = DrNurseMatchClassifier(llm=_fake_llm("MATCH", "Clinically consistent."))
    pair = _make_pair(
        _make_dr_meta(name="Alice A", dob="01/01/2020", dos="01/01/2026", sex="F"),
        _make_nurse_fields(name="Bob B", dob="02/02/2021", dos="02/02/2027", sex="Male"),
    )
    result = await clf.classify_pair(pair)
    assert result.overall == "MATCH"


# ---------------------------------------------------------------------------
# LLM error handling
# ---------------------------------------------------------------------------

async def test_llm_parse_error_returns_mismatch() -> None:
    clf = DrNurseMatchClassifier(llm=_invalid_llm())
    result = await clf.classify_pair(_make_pair(_make_dr_meta(), _make_nurse_fields()))
    assert result.clinical_verdict == "MISMATCH"
    assert result.clinical_reasoning  # some error message present


# ---------------------------------------------------------------------------
# Prompt input content
# ---------------------------------------------------------------------------

async def test_prompt_input_uses_dr_raw_text() -> None:
    clf = DrNurseMatchClassifier(llm=_fake_llm("MATCH"))
    dr = _make_dr_meta(raw_text="Fever and rash noted by physician.")
    prompt_input = clf._build_prompt_input(_make_pair(dr, _make_nurse_fields()))
    assert "Fever and rash" in prompt_input["dr_full_text"]


async def test_prompt_input_uses_nurse_raw_text() -> None:
    clf = DrNurseMatchClassifier(llm=_fake_llm("MATCH"))
    pair = _make_pair(
        _make_dr_meta(),
        _make_nurse_fields(),
        nurse_raw_text="High temperature and spots on skin observed.",
    )
    prompt_input = clf._build_prompt_input(pair)
    assert "High temperature" in prompt_input["nurse_full_text"]


# ---------------------------------------------------------------------------
# Diagnosis check (hospital pairs only)
# ---------------------------------------------------------------------------

def _fake_diagnosis_extractor(
    dr_diagnoses: list[Diagnosis] | None = None,
    nurse_diagnoses: list[Diagnosis] | None = None,
) -> MagicMock:
    """Extractor returning per-note diagnoses, keyed on the metadata's file path."""
    per_file = {
        "dr_note.pdf": dr_diagnoses or [],
        "nurse_note.pdf": nurse_diagnoses or [],
    }
    extractor = MagicMock()
    extractor.extract_diagnoses = AsyncMock(
        side_effect=lambda meta: DiagnosisExtractionResult(
            file_path=meta.file_path,
            diagnoses=per_file.get(meta.file_path.name, []),
        )
    )
    return extractor


async def test_diagnosis_check_exists_when_dr_note_has_diagnosis() -> None:
    extractor = _fake_diagnosis_extractor(dr_diagnoses=[Diagnosis(name="Colic", icd_code="R10.83")])
    clf = DrNurseMatchClassifier(llm=_fake_llm("MATCH"), diagnosis_extractor=extractor)
    result = await clf.classify_pair(_make_pair(_make_dr_meta(), _make_nurse_fields(), category="hospital"))
    assert result.diagnosis_check == DiagnosisCheck.EXISTS
    assert [(d.name, d.source) for d in result.diagnoses] == [("Colic", "dr")]
    assert result.overall == "MATCH"


async def test_diagnosis_check_exists_when_only_nurse_note_has_diagnosis() -> None:
    extractor = _fake_diagnosis_extractor(nurse_diagnoses=[Diagnosis(name="Otitis media")])
    clf = DrNurseMatchClassifier(llm=_fake_llm("MATCH"), diagnosis_extractor=extractor)
    result = await clf.classify_pair(_make_pair(_make_dr_meta(), _make_nurse_fields(), category="hospital"))
    assert result.diagnosis_check == DiagnosisCheck.EXISTS
    assert [(d.name, d.source) for d in result.diagnoses] == [("Otitis media", "nurse")]


async def test_diagnosis_check_collects_proof_from_both_notes() -> None:
    extractor = _fake_diagnosis_extractor(
        dr_diagnoses=[Diagnosis(name="Colic", icd_code="R10.83")],
        nurse_diagnoses=[Diagnosis(name="Reflux")],
    )
    clf = DrNurseMatchClassifier(llm=_fake_llm("MATCH"), diagnosis_extractor=extractor)
    result = await clf.classify_pair(_make_pair(_make_dr_meta(), _make_nurse_fields(), category="hospital"))
    assert [(d.name, d.source) for d in result.diagnoses] == [("Colic", "dr"), ("Reflux", "nurse")]


async def test_diagnosis_missing_forces_overall_mismatch() -> None:
    extractor = _fake_diagnosis_extractor()
    clf = DrNurseMatchClassifier(llm=_fake_llm("MATCH"), diagnosis_extractor=extractor)
    result = await clf.classify_pair(_make_pair(_make_dr_meta(), _make_nurse_fields(), category="hospital"))
    assert result.diagnosis_check == DiagnosisCheck.MISSING
    assert result.diagnoses == []
    assert result.clinical_verdict == "MATCH"  # clinical verdict itself untouched
    assert result.overall == "MISMATCH"


@pytest.mark.parametrize("category", [None, "peds"])
async def test_diagnosis_check_skipped_for_other_categories(category: str | None) -> None:
    extractor = _fake_diagnosis_extractor()
    clf = DrNurseMatchClassifier(llm=_fake_llm("MATCH"), diagnosis_extractor=extractor)
    result = await clf.classify_pair(_make_pair(_make_dr_meta(), _make_nurse_fields(), category=category))
    assert result.diagnosis_check is None
    assert result.diagnoses == []
    assert result.overall == "MATCH"
    extractor.extract_diagnoses.assert_not_awaited()


async def test_diagnosis_check_skipped_when_no_extractor_injected() -> None:
    clf = DrNurseMatchClassifier(llm=_fake_llm("MATCH"))
    result = await clf.classify_pair(_make_pair(_make_dr_meta(), _make_nurse_fields(), category="hospital"))
    assert result.diagnosis_check is None
    assert result.overall == "MATCH"


async def test_diagnosis_check_reads_nurse_raw_text() -> None:
    extractor = _fake_diagnosis_extractor(nurse_diagnoses=[Diagnosis(name="Reflux")])
    clf = DrNurseMatchClassifier(llm=_fake_llm("MATCH"), diagnosis_extractor=extractor)
    pair = _make_pair(_make_dr_meta(), _make_nurse_fields(), nurse_raw_text="Nurse text here", category="hospital")
    await clf.classify_pair(pair)
    texts = [call.args[0].raw_text for call in extractor.extract_diagnoses.await_args_list]
    assert "Nurse text here" in texts

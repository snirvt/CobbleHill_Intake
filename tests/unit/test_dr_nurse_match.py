import json
from pathlib import Path

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from classifier.classifiers.dr_nurse_match import DrNurseMatchClassifier
from classifier.models import (
    DocumentMetadata,
    MedicationData,
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
    assessment: list[str] | None = None,
    complaints: list[str] | None = None,
    plan: str = "Follow up in one month.",
) -> DocumentMetadata:
    return DocumentMetadata(
        file_path=Path("dr_note.pdf"),
        raw_text="",
        meta=PatientMetadata(
            patient_name=name,
            dob=dob,
            dos=dos,
            sex=sex,
            account_number=account,
        ),
        assessment=assessment or ["Colic - R10.83"],
        complaints=complaints or ["Crying", "Feeding issues"],
        plan=plan,
        medications=MedicationData(taking=[], not_taking=[]),
    )


def _make_nurse_fields(
    name: str = "Smith, John",
    dob: str = "01/15/2025",
    dos: str = "04/17/2026",
    sex: str = "Male",
    prn: str = "ABC123",
    diagnoses: list[str] | None = None,
    assessment: list[str] | None = None,
    chief_complaint: str = "Crying and feeding difficulty",
    plan: str = "Monitor feeding and weight.",
) -> NurseVisitFields:
    return NurseVisitFields(
        meta=NursePatientMeta(patient_name=name, dob=dob, dos=dos, sex=sex, prn=prn),
        diagnoses=diagnoses or ["(R10.83) Colic"],
        assessment=assessment or ["Infantile colic - R10.83"],
        chief_complaint=chief_complaint,
        plan=plan,
    )


def _make_pair(dr: DocumentMetadata, nurse: NurseVisitFields) -> PairDocumentMetadata:
    return PairDocumentMetadata(
        dr_file_path=Path("dr_note.pdf"),
        nurse_file_path=Path("nurse_note.pdf"),
        dr=dr,
        nurse=nurse,
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


async def test_clinical_verdict_partial_match() -> None:
    clf = DrNurseMatchClassifier(llm=_fake_llm("PARTIAL_MATCH", "Minor plan discrepancies."))
    result = await clf.classify_pair(_make_pair(_make_dr_meta(), _make_nurse_fields()))
    assert result.clinical_verdict == "PARTIAL_MATCH"


async def test_clinical_verdict_mismatch() -> None:
    clf = DrNurseMatchClassifier(llm=_fake_llm("MISMATCH", "Different diagnoses entirely."))
    result = await clf.classify_pair(_make_pair(_make_dr_meta(), _make_nurse_fields()))
    assert result.clinical_verdict == "MISMATCH"


# ---------------------------------------------------------------------------
# Overall verdict override logic
# ---------------------------------------------------------------------------

async def test_overall_mismatch_when_zero_identity_fields_match() -> None:
    clf = DrNurseMatchClassifier(llm=_fake_llm("MATCH", "Clinically consistent."))
    pair = _make_pair(
        _make_dr_meta(name="Alice A", dob="01/01/2020", dos="01/01/2026", sex="F"),
        _make_nurse_fields(name="Bob B", dob="02/02/2021", dos="02/02/2027", sex="Male"),
    )
    result = await clf.classify_pair(pair)
    assert result.overall == "MISMATCH"


async def test_overall_partial_match_when_few_identity_fields_match() -> None:
    clf = DrNurseMatchClassifier(llm=_fake_llm("MATCH", "Looks good."))
    pair = _make_pair(
        _make_dr_meta(name="Alice A", dob="01/01/2020", dos="04/17/2026", sex="F"),
        _make_nurse_fields(name="Bob B", dob="02/02/2021", dos="04/17/2026", sex="Female"),
    )
    result = await clf.classify_pair(pair)
    # Only dos and sex match (score=2), LLM says MATCH → demote to PARTIAL_MATCH
    assert result.overall == "PARTIAL_MATCH"


async def test_overall_propagates_mismatch_from_llm() -> None:
    clf = DrNurseMatchClassifier(llm=_fake_llm("MISMATCH", "Completely different cases."))
    result = await clf.classify_pair(_make_pair(_make_dr_meta(), _make_nurse_fields()))
    assert result.overall == "MISMATCH"


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

async def test_prompt_input_includes_dr_complaints() -> None:
    clf = DrNurseMatchClassifier(llm=_fake_llm("MATCH"))
    dr = _make_dr_meta(complaints=["Fever", "Rash"])
    prompt_input = clf._build_prompt_input(_make_pair(dr, _make_nurse_fields()))
    assert "Fever" in prompt_input["dr_complaints"]
    assert "Rash" in prompt_input["dr_complaints"]


async def test_prompt_input_includes_nurse_chief_complaint() -> None:
    clf = DrNurseMatchClassifier(llm=_fake_llm("MATCH"))
    nurse = _make_nurse_fields(chief_complaint="High temperature and spots on skin")
    prompt_input = clf._build_prompt_input(_make_pair(_make_dr_meta(), nurse))
    assert "High temperature" in prompt_input["nurse_complaint"]

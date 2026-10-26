from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from classifier.folder_checks import RequiredNotesCheck
from classifier.intake_review_pipeline import IntakeReviewPipeline
from classifier.metadata.nurse_visit import NurseVisitExtractor
from classifier.metadata.progress_note import ProgressNoteExtractor
from classifier.models import (
    CareCheck,
    ClassificationStepError,
    ClinicalVerdict,
    Diagnosis,
    DiagnosisCheck,
    DiagnosisExtractionResult,
    DocumentMetadata,
    DrValidityResult,
    ErrorCode,
    ErrorStage,
    ExtractedText,
    IntakeVerdict,
    PairDocumentMetadata,
    PatientGroup,
    PhaseVerdict,
    RunError,
    ValidityVerdict,
)

_GERD = Diagnosis(name="GERD", icd_code="K21.9")


def _touch(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _router(texts: dict[str, str] | None = None) -> MagicMock:
    """Extracts each file's text from ``texts`` by name ("note text" by default)."""
    texts = texts or {}

    async def extract(path: Path) -> ExtractedText:
        return ExtractedText(file_path=path, text=texts.get(path.name, "note text"), num_pages=1)

    extractor = MagicMock()
    extractor.extract = AsyncMock(side_effect=extract)
    router = MagicMock()
    router.route = MagicMock(return_value=extractor)
    return router


def _validity(
    group: PatientGroup = PatientGroup.ADULT,
    diagnosis: bool = True,
    care: CareCheck | None = None,
) -> MagicMock:
    async def classify(metadata: DocumentMetadata, dr_paths: list[Path]) -> DrValidityResult:
        return DrValidityResult(
            patient_folder=metadata.file_path.parent,
            dr_paths=dr_paths,
            patient_group=group,
            diagnosis_check=DiagnosisCheck.EXISTS if diagnosis else DiagnosisCheck.MISSING,
            diagnoses=[_GERD] if diagnosis else [],
            care_check=care,
            verdict=ValidityVerdict.VALID,
        )

    clf = MagicMock()
    clf.classify = AsyncMock(side_effect=classify)
    return clf


def _pair_clf(verdicts: dict[str | None, ClinicalVerdict] | None = None) -> MagicMock:
    """Clinical verdict per pair category (MATCH by default)."""
    verdicts = verdicts or {}

    async def classify_pair(pair: PairDocumentMetadata) -> MagicMock:
        result = MagicMock()
        result.clinical_verdict = verdicts.get(pair.category, ClinicalVerdict.MATCH)
        result.clinical_reasoning = "fake reasoning"
        return result

    clf = MagicMock()
    clf.classify_pair = AsyncMock(side_effect=classify_pair)
    return clf


def _nurse_dx(error: str | None = None) -> MagicMock:
    async def extract(metadata: DocumentMetadata) -> DiagnosisExtractionResult:
        return DiagnosisExtractionResult(
            file_path=metadata.file_path, diagnoses=[] if error else [_GERD], error=error
        )

    dx = MagicMock()
    dx.extract_diagnoses = AsyncMock(side_effect=extract)
    return dx


def _make(
    router: MagicMock | None = None,
    validity: MagicMock | None = None,
    pair_clf: MagicMock | None = None,
    nurse_dx: MagicMock | None = None,
) -> IntakeReviewPipeline:
    return IntakeReviewPipeline(
        router=router or _router(),
        validity_classifier=validity or _validity(),
        pair_classifier=pair_clf or _pair_clf(),
        nurse_diagnosis_extractor=nurse_dx or _nurse_dx(),
        dr_meta_extractor=ProgressNoteExtractor(),
        nurse_meta_extractor=NurseVisitExtractor(),
        folder_check=RequiredNotesCheck(roles=("dr",)),
    )


async def test_matching_pair_after_phase_1_pass_is_accepted(tmp_path: Path) -> None:
    dr = _touch(tmp_path / "P" / "dr_note.pdf")
    nurse = _touch(tmp_path / "P" / "nurse_visit.pdf")

    [r] = await _make().run_folder(tmp_path)

    assert r.success and r.result is not None
    assert r.nurse_paths == [nurse]
    assert r.result.verdict_phase_1 is PhaseVerdict.PASS
    assert r.result.verdict_phase_2 is PhaseVerdict.PASS
    assert r.result.verdict is IntakeVerdict.ACCEPT
    assert r.result.pairs[0].dr_paths == [dr]
    assert r.result.pairs[0].nurse_diagnoses == [_GERD]
    assert r.result.explanation == "Doctor and nurse both report GERD (K21.9)."


async def test_phase_1_failure_skips_phase_2_and_nurse_extraction(tmp_path: Path) -> None:
    _touch(tmp_path / "P" / "dr_note.pdf")
    _touch(tmp_path / "P" / "nurse_visit.pdf")
    router, pair_clf, nurse_dx = _router(), _pair_clf(), _nurse_dx()
    pipeline = _make(router, _validity(diagnosis=False), pair_clf, nurse_dx)

    [r] = await pipeline.run_folder(tmp_path)

    assert r.result is not None
    assert r.result.verdict_phase_1 is PhaseVerdict.FAIL
    assert r.result.verdict_phase_2 is PhaseVerdict.NOT_RUN
    assert r.result.verdict is IntakeVerdict.REJECT
    assert r.result.explanation == "Phase I failed: no valid diagnosis."
    pair_clf.classify_pair.assert_not_awaited()
    nurse_dx.extract_diagnoses.assert_not_awaited()
    routed = [c.args[0].name for c in router.route.call_args_list]
    assert routed == ["dr_note.pdf"]


async def test_pediatric_without_necessity_is_rejected(tmp_path: Path) -> None:
    _touch(tmp_path / "P" / "peds_dr_note.pdf")
    _touch(tmp_path / "P" / "peds_nurse_visit.pdf")
    validity = _validity(PatientGroup.PEDIATRIC, care=CareCheck.NOT_NEEDED)

    [r] = await _make(validity=validity).run_folder(tmp_path)

    assert r.result is not None and r.result.verdict is IntakeVerdict.REJECT
    assert r.result.phase_1_reasons == ["no statement of medical necessity"]


async def test_unknown_age_gets_warning_and_pediatric_rules(tmp_path: Path) -> None:
    _touch(tmp_path / "P" / "dr_note.pdf")
    _touch(tmp_path / "P" / "nurse_visit.pdf")
    validity = _validity(PatientGroup.UNKNOWN, care=CareCheck.NEEDED)

    [r] = await _make(validity=validity).run_folder(tmp_path)

    assert r.result is not None and r.result.verdict is IntakeVerdict.ACCEPT
    assert r.result.warnings == ["age not found — pediatric rules applied"]


async def test_no_nurse_note_has_no_phase_2_or_explanation(tmp_path: Path) -> None:
    _touch(tmp_path / "P" / "dr_note.pdf")

    [r] = await _make().run_folder(tmp_path)

    assert r.result is not None
    assert r.result.verdict_phase_1 is PhaseVerdict.PASS
    assert r.result.verdict_phase_2 is None
    assert r.result.verdict is IntakeVerdict.NO_NURSE_NOTE
    assert r.result.explanation is None


async def test_no_nurse_note_wins_over_phase_1_failure(tmp_path: Path) -> None:
    _touch(tmp_path / "P" / "dr_note.pdf")

    [r] = await _make(validity=_validity(diagnosis=False)).run_folder(tmp_path)

    assert r.result is not None
    assert r.result.verdict_phase_1 is PhaseVerdict.FAIL
    assert r.result.verdict is IntakeVerdict.NO_NURSE_NOTE


async def test_nurse_note_of_other_category_is_no_nurse_note(tmp_path: Path) -> None:
    _touch(tmp_path / "P" / "hospital_dr_note.pdf")
    _touch(tmp_path / "P" / "peds_nurse_visit.pdf")

    [r] = await _make().run_folder(tmp_path)

    assert r.result is not None and r.result.verdict is IntakeVerdict.NO_NURSE_NOTE


async def test_undecided_pair_needs_review(tmp_path: Path) -> None:
    _touch(tmp_path / "P" / "dr_note.pdf")
    _touch(tmp_path / "P" / "nurse_visit.pdf")
    pair_clf = _pair_clf({None: ClinicalVerdict.UNDECIDED})

    [r] = await _make(pair_clf=pair_clf).run_folder(tmp_path)

    assert r.result is not None
    assert r.result.verdict_phase_2 is PhaseVerdict.UNDECIDED
    assert r.result.verdict is IntakeVerdict.NEEDS_REVIEW


async def test_folders_without_dr_notes_are_listed_without_review(tmp_path: Path) -> None:
    nurse = _touch(tmp_path / "ONLY NURSE" / "nurse_visit.pdf")
    (tmp_path / "EMPTY").mkdir()
    _touch(tmp_path / "OK" / "dr_note.pdf")

    results = await _make().run_folder(tmp_path)

    assert [r.patient_folder.name for r in results] == ["OK", "EMPTY", "ONLY NURSE"]
    _, empty, only_nurse = results
    assert empty.success and empty.result is None and empty.nurse_paths == []
    assert only_nurse.success and only_nurse.result is None
    assert only_nurse.dr_paths == [] and only_nurse.nurse_paths == [nurse]


async def test_one_mismatching_pair_rejects_patient(tmp_path: Path) -> None:
    for name in ("hospital_dr_a.pdf", "hospital_nurse_a.pdf", "peds_dr_b.pdf", "peds_nurse_b.pdf"):
        _touch(tmp_path / "P" / name)
    pair_clf = _pair_clf({"peds": ClinicalVerdict.MISMATCH})

    [r] = await _make(pair_clf=pair_clf).run_folder(tmp_path)

    assert r.result is not None
    assert [p.category for p in r.result.pairs] == ["hospital", "peds"]
    assert r.result.verdict_phase_2 is PhaseVerdict.FAIL
    assert r.result.verdict is IntakeVerdict.REJECT
    assert r.result.explanation is not None
    assert r.result.explanation.startswith("[peds] Notes do not match")


async def test_nurse_diagnosis_failure_is_reported(tmp_path: Path) -> None:
    _touch(tmp_path / "P" / "dr_note.pdf")
    _touch(tmp_path / "P" / "nurse_visit.pdf")

    [r] = await _make(nurse_dx=_nurse_dx(error="LLM error: timeout")).run_folder(tmp_path)

    assert not r.success
    assert [(e.stage, e.code) for e in r.errors] == [(ErrorStage.DIAGNOSIS, ErrorCode.LLM_FAILED)]


async def test_classifier_step_error_is_reported(tmp_path: Path) -> None:
    _touch(tmp_path / "P" / "dr_note.pdf")
    step_error = RunError(
        stage=ErrorStage.AGE, code=ErrorCode.LLM_FAILED, message="LLM error: x"
    )
    validity = MagicMock()
    validity.classify = AsyncMock(side_effect=ClassificationStepError([step_error]))

    [r] = await _make(validity=validity).run_folder(tmp_path)

    assert not r.success and r.errors == [step_error]


async def test_unexpected_error_is_reported_without_aborting_batch(tmp_path: Path) -> None:
    _touch(tmp_path / "BAD" / "dr_note.pdf")
    _touch(tmp_path / "GOOD" / "dr_note.pdf")

    async def classify(metadata: DocumentMetadata, dr_paths: list[Path]) -> DrValidityResult:
        if metadata.file_path.parent.name == "BAD":
            raise RuntimeError("LiteParse failed")
        return await _validity().classify(metadata, dr_paths)

    validity = MagicMock()
    validity.classify = AsyncMock(side_effect=classify)

    bad, good = await _make(validity=validity).run_folder(tmp_path)

    assert not bad.success and bad.errors[0].code is ErrorCode.UNEXPECTED
    assert good.success


async def test_blank_dr_notes_fail_without_classifying(tmp_path: Path) -> None:
    _touch(tmp_path / "P" / "dr_note.pdf")
    validity = _validity()

    [r] = await _make(_router({"dr_note.pdf": "  "}), validity).run_folder(tmp_path)

    assert not r.success
    assert r.errors[0].code is ErrorCode.EMPTY_TEXT
    validity.classify.assert_not_awaited()


async def test_blank_nurse_note_fails_patient(tmp_path: Path) -> None:
    _touch(tmp_path / "P" / "dr_note.pdf")
    _touch(tmp_path / "P" / "nurse_visit.pdf")
    pair_clf = _pair_clf()

    [r] = await _make(_router({"nurse_visit.pdf": ""}), pair_clf=pair_clf).run_folder(tmp_path)

    assert not r.success
    assert [e.file.name for e in r.errors if e.file] == ["nurse_visit.pdf"]
    pair_clf.classify_pair.assert_not_awaited()


def test_folder_errors_report_folders_without_dr_notes(tmp_path: Path) -> None:
    _touch(tmp_path / "ONLY NURSE" / "nurse_visit.pdf")
    _touch(tmp_path / "OK" / "dr_note.pdf")

    errors = _make().folder_errors(tmp_path)

    assert [e.folder.name for e in errors if e.folder] == ["ONLY NURSE"]
    assert errors[0].code is ErrorCode.MISSING_REQUIRED_NOTES

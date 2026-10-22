from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from classifier.folder_checks import RequiredNotesCheck
from classifier.models import (
    ClassificationStepError,
    DocumentMetadata,
    ErrorCode,
    ErrorStage,
    ExtractedFields,
    ExtractedText,
    IdentityMatchResult,
    NursePatientMeta,
    NurseVisitFields,
    PairClassificationResult,
    PatientMetadata,
    RunError,
)
from classifier.pair_pipeline import PairPipeline, scan_pairs

# ---------------------------------------------------------------------------
# scan_pairs
# ---------------------------------------------------------------------------

def test_scan_pairs_finds_pair_in_root(tmp_path: Path) -> None:
    (tmp_path / "dr_note.pdf").write_bytes(b"%PDF")
    (tmp_path / "nurse_visit.pdf").write_bytes(b"%PDF")
    pairs = scan_pairs(tmp_path)
    assert len(pairs) == 1
    assert [p.name for p in pairs[0][0]] == ["dr_note.pdf"]
    assert [p.name for p in pairs[0][1]] == ["nurse_visit.pdf"]


def test_scan_pairs_finds_pairs_in_subfolders(tmp_path: Path) -> None:
    sub1 = tmp_path / "patient_a"
    sub2 = tmp_path / "patient_b"
    sub1.mkdir()
    sub2.mkdir()
    (sub1 / "dr_progress.pdf").write_bytes(b"%PDF")
    (sub1 / "nurse_visit.pdf").write_bytes(b"%PDF")
    (sub2 / "dr_note.pdf").write_bytes(b"%PDF")
    (sub2 / "nurse_note.pdf").write_bytes(b"%PDF")
    pairs = scan_pairs(tmp_path)
    assert len(pairs) == 2


def test_scan_pairs_skips_folder_missing_dr(tmp_path: Path) -> None:
    sub = tmp_path / "incomplete"
    sub.mkdir()
    (sub / "nurse_visit.pdf").write_bytes(b"%PDF")
    pairs = scan_pairs(tmp_path)
    assert pairs == []


def test_scan_pairs_skips_folder_missing_nurse(tmp_path: Path) -> None:
    sub = tmp_path / "incomplete"
    sub.mkdir()
    (sub / "dr_note.pdf").write_bytes(b"%PDF")
    pairs = scan_pairs(tmp_path)
    assert pairs == []


def test_scan_pairs_includes_all_notes_sorted(tmp_path: Path) -> None:
    (tmp_path / "dr_aaa.pdf").write_bytes(b"%PDF")
    (tmp_path / "dr_zzz.pdf").write_bytes(b"%PDF")
    (tmp_path / "nurse_aaa.pdf").write_bytes(b"%PDF")
    pairs = scan_pairs(tmp_path)
    assert len(pairs) == 1
    assert [p.name for p in pairs[0][0]] == ["dr_aaa.pdf", "dr_zzz.pdf"]
    assert [p.name for p in pairs[0][1]] == ["nurse_aaa.pdf"]


def test_scan_pairs_matches_category_prefixed_notes(tmp_path: Path) -> None:
    (tmp_path / "hospital_dr_note.pdf").write_bytes(b"%PDF")
    (tmp_path / "hospital_nurse_visit.pdf").write_bytes(b"%PDF")
    pairs = scan_pairs(tmp_path)
    assert len(pairs) == 1
    assert [p.name for p in pairs[0][0]] == ["hospital_dr_note.pdf"]
    assert [p.name for p in pairs[0][1]] == ["hospital_nurse_visit.pdf"]


def test_scan_pairs_separates_categories_in_one_folder(tmp_path: Path) -> None:
    (tmp_path / "hospital_dr_a.pdf").write_bytes(b"%PDF")
    (tmp_path / "hospital_nurse_a.pdf").write_bytes(b"%PDF")
    (tmp_path / "peds_dr_b.pdf").write_bytes(b"%PDF")
    (tmp_path / "peds_nurse_b.pdf").write_bytes(b"%PDF")
    (tmp_path / "dr_c.pdf").write_bytes(b"%PDF")
    (tmp_path / "nurse_c.pdf").write_bytes(b"%PDF")
    pairs = scan_pairs(tmp_path)
    assert [([d.name for d in dr], [n.name for n in nurse]) for dr, nurse in pairs] == [
        (["dr_c.pdf"], ["nurse_c.pdf"]),
        (["hospital_dr_a.pdf"], ["hospital_nurse_a.pdf"]),
        (["peds_dr_b.pdf"], ["peds_nurse_b.pdf"]),
    ]


def test_scan_pairs_does_not_cross_pair_categories(tmp_path: Path) -> None:
    (tmp_path / "hospital_dr_a.pdf").write_bytes(b"%PDF")
    (tmp_path / "peds_nurse_b.pdf").write_bytes(b"%PDF")
    assert scan_pairs(tmp_path) == []


def test_scan_pairs_appends_multiple_notes_within_a_category(tmp_path: Path) -> None:
    (tmp_path / "peds_dr_zzz.pdf").write_bytes(b"%PDF")
    (tmp_path / "peds_dr_aaa.pdf").write_bytes(b"%PDF")
    (tmp_path / "peds_nurse_a.pdf").write_bytes(b"%PDF")
    pairs = scan_pairs(tmp_path)
    assert len(pairs) == 1
    assert [p.name for p in pairs[0][0]] == ["peds_dr_aaa.pdf", "peds_dr_zzz.pdf"]


def test_scan_pairs_ignores_category_prefix_without_role(tmp_path: Path) -> None:
    (tmp_path / "peds_summary.pdf").write_bytes(b"%PDF")
    (tmp_path / "peds_nurse_visit.pdf").write_bytes(b"%PDF")
    assert scan_pairs(tmp_path) == []


def test_scan_pairs_ignores_unsupported_extensions(tmp_path: Path) -> None:
    (tmp_path / "dr_note.txt").write_bytes(b"text")
    (tmp_path / "nurse_visit.txt").write_bytes(b"text")
    pairs = scan_pairs(tmp_path)
    assert pairs == []


# ---------------------------------------------------------------------------
# PairPipeline.run_pairs
# ---------------------------------------------------------------------------

_FOLDER_CHECK = RequiredNotesCheck(roles=("dr", "nurse"), same_category=True)


def _make_pair_pipeline(
    pair_result: PairClassificationResult,
    dr_text: str = "dr note text",
    nurse_text: str = "nurse note text",
) -> PairPipeline:
    extractor = MagicMock()
    extractor.extract = AsyncMock(
        side_effect=lambda p: ExtractedText(file_path=p, text=dr_text if "dr" in p.name else nurse_text, num_pages=1)
    )

    router = MagicMock()
    router.route = MagicMock(return_value=extractor)

    dr_meta_extractor = MagicMock()
    dr_meta_extractor.extract = MagicMock(
        return_value=ExtractedFields(
            meta=PatientMetadata(patient_name="Test", dob="01/01/2025", dos="04/17/2026", sex="F", account_number="X1"),
        )
    )

    nurse_meta_extractor = MagicMock()
    nurse_meta_extractor.extract = MagicMock(return_value=NurseVisitFields(meta=NursePatientMeta(patient_name="Test")))

    pair_clf = MagicMock()
    pair_clf.classify_pair = AsyncMock(return_value=pair_result)

    return PairPipeline(
        router=router,
        dr_meta_extractor=dr_meta_extractor,
        nurse_meta_extractor=nurse_meta_extractor,
        pair_classifier=pair_clf,
        folder_check=_FOLDER_CHECK,
    )


def _sample_pair_result(dr_path: Path, nurse_path: Path) -> PairClassificationResult:
    dr_meta = DocumentMetadata(
        file_path=dr_path,
        raw_text="dr note text",
        meta=PatientMetadata(patient_name="Test"),
    )
    return PairClassificationResult(
        dr_file_path=dr_path,
        nurse_file_path=nurse_path,
        identity_match=IdentityMatchResult(patient_name=True, dob=True, dos=True, sex=True),
        clinical_verdict="MATCH",
        clinical_reasoning="All consistent.",
        overall="MATCH",
        dr_metadata=dr_meta,
        nurse_fields=NurseVisitFields(meta=NursePatientMeta(patient_name="Test")),
    )


async def test_pair_pipeline_returns_success_result(tmp_path: Path) -> None:
    dr = tmp_path / "dr_note.pdf"
    nurse = tmp_path / "nurse_visit.pdf"
    dr.write_bytes(b"%PDF")
    nurse.write_bytes(b"%PDF")

    pair_result = _sample_pair_result(dr, nurse)
    pipeline = _make_pair_pipeline(pair_result)
    results = await pipeline.run_pairs([([dr], [nurse])])

    assert len(results) == 1
    assert results[0].success is True
    assert results[0].result is not None
    assert results[0].result.overall == "MATCH"


async def test_pair_pipeline_appends_multiple_notes_with_headers(tmp_path: Path) -> None:
    dr_a = tmp_path / "dr_note_a.pdf"
    dr_b = tmp_path / "dr_note_b.pdf"
    nurse = tmp_path / "nurse_visit.pdf"
    for f in (dr_a, dr_b, nurse):
        f.write_bytes(b"%PDF")

    # Distinct text per file so we can assert both are appended.
    extractor = MagicMock()
    extractor.extract = AsyncMock(
        side_effect=lambda p: ExtractedText(file_path=p, text=f"TEXT[{p.name}]", num_pages=1)
    )
    router = MagicMock()
    router.route = MagicMock(return_value=extractor)

    dr_meta_extractor = MagicMock()
    dr_meta_extractor.extract = MagicMock(
        return_value=ExtractedFields(meta=PatientMetadata(patient_name="Test"))
    )
    nurse_meta_extractor = MagicMock()
    nurse_meta_extractor.extract = MagicMock(
        return_value=NurseVisitFields(meta=NursePatientMeta(patient_name="Test"))
    )
    pair_clf = MagicMock()
    pair_clf.classify_pair = AsyncMock(
        return_value=_sample_pair_result(dr_a, nurse)
    )

    pipeline = PairPipeline(
        router=router,
        dr_meta_extractor=dr_meta_extractor,
        nurse_meta_extractor=nurse_meta_extractor,
        pair_classifier=pair_clf,
        folder_check=_FOLDER_CHECK,
    )
    results = await pipeline.run_pairs([([dr_a, dr_b], [nurse])])

    assert results[0].success is True
    assert [p.name for p in results[0].dr_paths] == ["dr_note_a.pdf", "dr_note_b.pdf"]
    dr_text_seen = dr_meta_extractor.extract.call_args.args[0]
    assert "--- dr_note_a.pdf ---" in dr_text_seen
    assert "--- dr_note_b.pdf ---" in dr_text_seen
    assert "TEXT[dr_note_a.pdf]" in dr_text_seen
    assert "TEXT[dr_note_b.pdf]" in dr_text_seen


async def test_pair_pipeline_returns_failure_on_extractor_error(tmp_path: Path) -> None:
    dr = tmp_path / "dr_note.pdf"
    nurse = tmp_path / "nurse_visit.pdf"
    dr.write_bytes(b"%PDF")
    nurse.write_bytes(b"%PDF")

    extractor = MagicMock()
    extractor.extract = AsyncMock(side_effect=RuntimeError("parse error"))
    router = MagicMock()
    router.route = MagicMock(return_value=extractor)

    pipeline = PairPipeline(
        router=router,
        dr_meta_extractor=MagicMock(),
        nurse_meta_extractor=MagicMock(),
        pair_classifier=MagicMock(),
        folder_check=_FOLDER_CHECK,
    )
    results = await pipeline.run_pairs([([dr], [nurse])])
    assert results[0].success is False
    assert "parse error" in (results[0].error or "")


async def test_pair_pipeline_run_folder_finds_pairs(tmp_path: Path) -> None:
    dr = tmp_path / "dr_progress.pdf"
    nurse = tmp_path / "nurse_visit.pdf"
    dr.write_bytes(b"%PDF")
    nurse.write_bytes(b"%PDF")

    pair_result = _sample_pair_result(dr, nurse)
    pipeline = _make_pair_pipeline(pair_result)
    results = await pipeline.run_folder(tmp_path)
    assert len(results) == 1
    assert results[0].success is True


async def test_pair_pipeline_run_folder_returns_empty_when_no_pairs(tmp_path: Path) -> None:
    pipeline = _make_pair_pipeline(_sample_pair_result(tmp_path / "a.pdf", tmp_path / "b.pdf"))
    results = await pipeline.run_folder(tmp_path)
    assert results == []


# ---------------------------------------------------------------------------
# Category propagation to the pair classifier
# ---------------------------------------------------------------------------

async def _classified_pair_metadata(tmp_path: Path, dr_name: str, nurse_name: str):  # type: ignore[no-untyped-def]
    """Run the pipeline on one pair and return the metadata handed to the classifier."""
    dr, nurse = tmp_path / dr_name, tmp_path / nurse_name
    dr.write_bytes(b"%PDF")
    nurse.write_bytes(b"%PDF")
    pipeline = _make_pair_pipeline(_sample_pair_result(dr, nurse))
    await pipeline.run_pairs([([dr], [nurse])])
    return pipeline._pair_clf.classify_pair.await_args.args[0]


async def test_pair_pipeline_passes_category_to_classifier(tmp_path: Path) -> None:
    pair = await _classified_pair_metadata(tmp_path, "hospital_dr_note.pdf", "hospital_nurse_visit.pdf")
    assert pair.category == "hospital"


async def test_pair_pipeline_passes_none_category_for_uncategorized_notes(tmp_path: Path) -> None:
    pair = await _classified_pair_metadata(tmp_path, "dr_note.pdf", "nurse_visit.pdf")
    assert pair.category is None


# ---------------------------------------------------------------------------
# Structured errors
# ---------------------------------------------------------------------------

def _write_pair(
    folder: Path, dr: str = "dr_note.pdf", nurse: str = "nurse_visit.pdf"
) -> tuple[Path, Path]:
    folder.mkdir(parents=True, exist_ok=True)
    dr_path, nurse_path = folder / dr, folder / nurse
    dr_path.write_bytes(b"%PDF")
    nurse_path.write_bytes(b"%PDF")
    return dr_path, nurse_path


async def test_pair_pipeline_fails_without_llm_when_one_side_blank(tmp_path: Path) -> None:
    dr, nurse = _write_pair(tmp_path)
    pipeline = _make_pair_pipeline(_sample_pair_result(dr, nurse), nurse_text="   ")

    [result] = await pipeline.run_pairs([([dr], [nurse])])

    assert result.success is False
    assert [(e.stage, e.code, e.file) for e in result.errors] == [
        (ErrorStage.EXTRACT, ErrorCode.EMPTY_TEXT, nurse)
    ]
    pipeline._pair_clf.classify_pair.assert_not_awaited()


async def test_pair_pipeline_keeps_blank_note_as_non_fatal_error(tmp_path: Path) -> None:
    dr_a, nurse = _write_pair(tmp_path, dr="dr_a.pdf")
    dr_b = tmp_path / "dr_b.pdf"
    dr_b.write_bytes(b"%PDF")
    pipeline = _make_pair_pipeline(_sample_pair_result(dr_a, nurse))
    extractor = pipeline._router.route.return_value
    extractor.extract.side_effect = lambda p: ExtractedText(
        file_path=p, text="" if p == dr_b else "text", num_pages=1
    )

    [result] = await pipeline.run_pairs([([dr_a, dr_b], [nurse])])

    assert result.success is True
    assert [e.file for e in result.errors] == [dr_b]


async def test_pair_pipeline_reports_classifier_step_errors(tmp_path: Path) -> None:
    dr, nurse = _write_pair(tmp_path)
    pipeline = _make_pair_pipeline(_sample_pair_result(dr, nurse))
    step_error = RunError(
        stage=ErrorStage.CLINICAL_MATCH, code=ErrorCode.LLM_FAILED, message="LLM error: down"
    )
    pipeline._pair_clf.classify_pair.side_effect = ClassificationStepError([step_error])

    [result] = await pipeline.run_pairs([([dr], [nurse])])

    assert result.success is False
    assert result.errors == [step_error]
    assert result.error == "clinical_match: LLM error: down"


async def test_pair_pipeline_reports_unexpected_error(tmp_path: Path) -> None:
    dr, nurse = _write_pair(tmp_path)
    pipeline = _make_pair_pipeline(_sample_pair_result(dr, nurse))
    pipeline._pair_clf.classify_pair.side_effect = RuntimeError("boom")

    [result] = await pipeline.run_pairs([([dr], [nurse])])

    assert result.success is False
    [error] = result.errors
    assert (error.stage, error.code, error.message, error.folder) == (
        ErrorStage.PIPELINE, ErrorCode.UNEXPECTED, "boom", tmp_path
    )


def test_scan_pairs_ignores_results_folders(tmp_path: Path) -> None:
    from config.settings import settings

    _write_pair(tmp_path / "patient")
    _write_pair(tmp_path / "patient" / settings.sharepoint_results_subfolder)

    pairs = scan_pairs(tmp_path)

    assert [p[0][0].parent.name for p in pairs] == ["patient"]


def test_folder_errors_reports_folders_without_a_pair(tmp_path: Path) -> None:
    _write_pair(tmp_path / "ok")
    (tmp_path / "dr_only").mkdir()
    (tmp_path / "dr_only" / "dr_note.pdf").write_bytes(b"%PDF")
    (tmp_path / "nurse_only").mkdir()
    (tmp_path / "nurse_only" / "nurse_note.pdf").write_bytes(b"%PDF")
    _write_pair(tmp_path / "mixed", dr="hospital_dr_note.pdf", nurse="peds_nurse_visit.pdf")
    (tmp_path / "empty").mkdir()
    pipeline = _make_pair_pipeline(_sample_pair_result(tmp_path / "a.pdf", tmp_path / "b.pdf"))

    errors = pipeline.folder_errors(tmp_path)

    assert [(e.folder.name if e.folder else None, e.code, e.message) for e in errors] == [
        ("dr_only", ErrorCode.MISSING_REQUIRED_NOTES, "Folder has no nurse notes"),
        ("empty", ErrorCode.EMPTY_FOLDER, "Folder has no files"),
        (
            "mixed",
            ErrorCode.MISSING_REQUIRED_NOTES,
            "Folder has no dr and nurse notes of the same category",
        ),
        ("nurse_only", ErrorCode.MISSING_REQUIRED_NOTES, "Folder has no dr notes"),
    ]


async def test_injected_semaphore_caps_pairs_across_concurrent_runs(tmp_path: Path) -> None:
    import asyncio

    pairs = [_write_pair(tmp_path / f"p{i}") for i in range(4)]
    pipeline = _make_pair_pipeline(_sample_pair_result(*pairs[0]))
    pipeline._semaphore = asyncio.Semaphore(1)
    active = peak = 0

    async def classify(pair: object) -> PairClassificationResult:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return _sample_pair_result(*pairs[0])

    pipeline._pair_clf.classify_pair.side_effect = classify

    await asyncio.gather(
        pipeline.run_pairs([([d], [n]) for d, n in pairs[:2]]),
        pipeline.run_pairs([([d], [n]) for d, n in pairs[2:]]),
    )

    assert peak == 1


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"%PDF")
    return path


def test_pairs_in_folder_does_not_recurse(tmp_path: Path) -> None:
    from classifier.pair_pipeline import pairs_in_folder

    dr = _touch(tmp_path / "dr_a.pdf")
    nurse = _touch(tmp_path / "nurse_a.pdf")
    _touch(tmp_path / "SUB" / "dr_b.pdf")
    _touch(tmp_path / "SUB" / "nurse_b.pdf")

    assert pairs_in_folder(tmp_path) == [([dr], [nurse])]


def test_build_pair_metadata_combines_texts_and_reads_category(tmp_path: Path) -> None:
    from classifier.metadata.nurse_visit import NurseVisitExtractor
    from classifier.metadata.progress_note import ProgressNoteExtractor
    from classifier.pair_pipeline import build_pair_metadata

    dr = _touch(tmp_path / "peds_dr_a.pdf")
    nurse = _touch(tmp_path / "peds_nurse_a.pdf")

    pair = build_pair_metadata(
        ProgressNoteExtractor(), NurseVisitExtractor(), [dr], [nurse], ["dr text"], ["nurse text"]
    )

    assert pair.category == "peds"
    assert pair.dr.raw_text == "--- peds_dr_a.pdf ---\ndr text"
    assert pair.nurse_raw_text == "--- peds_nurse_a.pdf ---\nnurse text"

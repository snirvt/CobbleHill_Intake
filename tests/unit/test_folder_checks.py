from pathlib import Path

from classifier.folder_checks import RequiredNotesCheck, leaf_folders
from classifier.models import ErrorCode, ErrorStage


def _touch(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


# ---------------------------------------------------------------------------
# leaf_folders
# ---------------------------------------------------------------------------

def test_leaf_folders_returns_only_folders_without_subfolders(tmp_path: Path) -> None:
    _touch(tmp_path / "Pending" / "DOE, JANE" / "dr_a.txt")
    _touch(tmp_path / "Pending" / "note_at_parent.txt")
    (tmp_path / "Pending" / "EMPTY").mkdir()
    (tmp_path / "Other").mkdir()

    assert leaf_folders(tmp_path) == [
        tmp_path / "Other",
        tmp_path / "Pending" / "DOE, JANE",
        tmp_path / "Pending" / "EMPTY",
    ]


def test_leaf_folders_patient_with_only_results_subfolder_is_a_leaf(tmp_path: Path) -> None:
    _touch(tmp_path / "dr_a.txt")
    (tmp_path / "results").mkdir()
    assert leaf_folders(tmp_path) == [tmp_path]


def test_leaf_folders_ignores_results_folders_at_any_depth(tmp_path: Path) -> None:
    _touch(tmp_path / "DOE, JANE" / "dr_a.txt")
    (tmp_path / "DOE, JANE" / "RESULTS").mkdir()
    (tmp_path / "results" / "old").mkdir(parents=True)

    assert leaf_folders(tmp_path) == [tmp_path / "DOE, JANE"]


def test_leaf_folders_returns_root_when_it_has_no_subfolders(tmp_path: Path) -> None:
    _touch(tmp_path / "dr_a.txt")
    assert leaf_folders(tmp_path) == [tmp_path]


# ---------------------------------------------------------------------------
# RequiredNotesCheck
# ---------------------------------------------------------------------------

def test_required_notes_check_passes_folder_with_dr_note(tmp_path: Path) -> None:
    _touch(tmp_path / "dr_a.txt")
    _touch(tmp_path / "nurse_visit.txt")
    assert RequiredNotesCheck(roles=("dr",)).check(tmp_path) is None


def test_required_notes_check_reports_empty_folder(tmp_path: Path) -> None:
    error = RequiredNotesCheck(roles=("dr",)).check(tmp_path)

    assert error is not None
    assert (error.stage, error.code, error.folder) == (
        ErrorStage.FOLDER,
        ErrorCode.EMPTY_FOLDER,
        tmp_path,
    )


def test_required_notes_check_reports_folder_without_dr_notes(tmp_path: Path) -> None:
    _touch(tmp_path / "nurse_visit.txt")
    _touch(tmp_path / "dr_note.docx")  # unsupported extension: not a note

    error = RequiredNotesCheck(roles=("dr",)).check(tmp_path)

    assert error is not None
    assert error.code is ErrorCode.MISSING_REQUIRED_NOTES
    assert error.message == "Folder has no dr notes"


def test_required_notes_check_roles_are_configurable(tmp_path: Path) -> None:
    _touch(tmp_path / "dr_a.txt")
    both = RequiredNotesCheck(roles=("dr", "nurse"))

    error = both.check(tmp_path)
    assert error is not None and error.message == "Folder has no nurse notes"

    _touch(tmp_path / "nurse_visit.txt")
    assert both.check(tmp_path) is None

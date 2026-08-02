from pathlib import Path

import pytest

from classifier.naming import NoteName, category_order, is_supported_file, label, parse_note, parse_note_name


# ---------------------------------------------------------------------------
# parse_note_name
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("dr_progress_note.pdf", NoteName(role="dr")),
        ("nurse_visit.txt", NoteName(role="nurse")),
        ("hospital_dr_note.pdf", NoteName(role="dr", category="hospital")),
        ("hospital_nurse_visit.pdf", NoteName(role="nurse", category="hospital")),
        ("peds_dr_note.pdf", NoteName(role="dr", category="peds")),
        ("peds_nurse_visit.txt", NoteName(role="nurse", category="peds")),
    ],
)
def test_parse_note_name_returns_role_and_optional_category(name: str, expected: NoteName) -> None:
    assert parse_note_name(name) == expected


def test_parse_note_name_is_case_insensitive() -> None:
    assert parse_note_name("Hospital_Nurse_Visit.PDF") == NoteName(role="nurse", category="hospital")


@pytest.mark.parametrize(
    "name",
    [
        "summary.pdf",         # no prefix at all
        "peds_summary.pdf",    # category without role
        "hospital_visit.txt",  # category without role
        "drnote.pdf",          # role word without separator
    ],
)
def test_parse_note_name_rejects_non_notes(name: str) -> None:
    assert parse_note_name(name) is None


def test_parse_note_name_role_before_category_keeps_no_category() -> None:
    assert parse_note_name("nurse_hospital_visit.pdf") == NoteName(role="nurse")


# ---------------------------------------------------------------------------
# parse_note / is_supported_file
# ---------------------------------------------------------------------------

def test_parse_note_rejects_unsupported_extension(tmp_path: Path) -> None:
    f = tmp_path / "hospital_dr_note.docx"
    f.write_text("x")
    assert parse_note(f) is None


def test_parse_note_filters_by_role(tmp_path: Path) -> None:
    f = tmp_path / "peds_nurse_visit.txt"
    f.write_text("x")
    assert parse_note(f, role="nurse") == NoteName(role="nurse", category="peds")
    assert parse_note(f, role="dr") is None


def test_parse_note_rejects_directory(tmp_path: Path) -> None:
    d = tmp_path / "dr_folder.pdf"
    d.mkdir()
    assert parse_note(d) is None


def test_is_supported_file_true_for_supported_extension(tmp_path: Path) -> None:
    f = tmp_path / "anything.pdf"
    f.write_bytes(b"%PDF")
    assert is_supported_file(f) is True


# ---------------------------------------------------------------------------
# category_order / label
# ---------------------------------------------------------------------------

def test_category_order_puts_uncategorized_first() -> None:
    order = category_order()
    assert order[0] is None
    assert "hospital" in order and "peds" in order


def test_label_names_uncategorized() -> None:
    assert label(None) == "uncategorized"
    assert label("peds") == "peds"

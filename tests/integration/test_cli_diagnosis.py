"""Integration test for the --diagnosis CLI flag.

Uses txt notes only (plaintext extractor — no npm/OCR) and stub mode (deterministic
LLM output), so it needs no external services. Marked integration; skipped by default.

Run with: pytest -m integration
"""

import json
import subprocess
import sys
from pathlib import Path

import openpyxl
import pytest

_REPO_ROOT = Path(__file__).parents[2]


@pytest.mark.integration
def test_cli_diagnosis_processes_dr_notes_only(tmp_path: Path) -> None:
    data = tmp_path / "data"
    (data / "1").mkdir(parents=True)
    (data / "1" / "dr_progress_note.txt").write_text("Assessment: some condition\n")
    (data / "1" / "nurse_visit.txt").write_text("Nurse note text\n")

    result = subprocess.run(
        [sys.executable, "-m", "cli", "--input", str(data), "--diagnosis"],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
    )

    assert result.returncode == 0, f"CLI failed:\n{result.stderr}"

    # JSON: only the dr_ note, nurse ignored
    output = json.loads(result.stdout)
    assert len(output) == 1
    assert output[0]["file"].endswith("dr_progress_note.txt")
    assert output[0]["diagnoses"] == [
        {"name": "STUB_DIAGNOSIS", "icd_code": "Z00.0", "source": None}
    ]

    # xlsx written next to configured output path
    xlsx_path = _REPO_ROOT / "output" / "diagnosis_results.xlsx"
    assert xlsx_path.exists()
    wb = openpyxl.load_workbook(xlsx_path)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))  # type: ignore[union-attr]
    assert rows[0] == ("file_path", "diagnosis", "icd_code")
    assert any(
        r[1] == "STUB_DIAGNOSIS" and str(r[0]).endswith("dr_progress_note.txt")
        for r in rows[1:]
    )

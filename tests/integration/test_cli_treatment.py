"""Integration test for the --treatment-request CLI flag.

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
def test_cli_treatment_request_processes_dr_notes_only(tmp_path: Path) -> None:
    data = tmp_path / "data"
    (data / "1").mkdir(parents=True)
    (data / "1" / "dr_progress_note.txt").write_text("Patient requested antibiotics.\n")
    (data / "1" / "nurse_visit.txt").write_text("Nurse note text\n")

    result = subprocess.run(
        [sys.executable, "-m", "cli", "--input", str(data), "--treatment-request"],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
    )

    assert result.returncode == 0, f"CLI failed:\n{result.stderr}"

    output = json.loads(result.stdout)
    assert len(output) == 1
    assert output[0]["file"].endswith("dr_progress_note.txt")
    # stub mode → deterministic False
    assert output[0]["treatment_requested"] is False
    assert "STUB" in output[0]["reasoning"]

    xlsx_path = _REPO_ROOT / "output" / "treatment_request_results.xlsx"
    assert xlsx_path.exists()
    wb = openpyxl.load_workbook(xlsx_path)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))  # type: ignore[union-attr]
    assert rows[0] == ("file_path", "treatment_requested", "reasoning")
    assert any(str(r[0]).endswith("dr_progress_note.txt") for r in rows[1:])

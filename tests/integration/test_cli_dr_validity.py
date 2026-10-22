"""Integration test for the --dr-validity CLI flag.

Uses txt notes only (plaintext extractor — no OCR), but hits the configured ollama
model, so it needs that service running. Assertions are model-agnostic: they check
grouping and output shape, not the verdict itself.

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
def test_cli_dr_validity_one_row_per_patient(tmp_path: Path) -> None:
    patient = tmp_path / "Pending" / "DOE, JANE"
    patient.mkdir(parents=True)
    (patient / "dr_DOE, JANE note.txt").write_text(
        "Patient: Jane Doe (8 mo F)\nAssessment: Infantile colic - R10.83\n"
    )
    (patient / "dr_necessity.txt").write_text(
        "Statement of medical necessity: patient requires skilled home nursing services.\n"
    )
    (patient / "nurse_visit.txt").write_text("Nurse note text\n")

    result = subprocess.run(
        [sys.executable, "-m", "cli", "--input", str(tmp_path / "Pending"), "--dr-validity"],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
    )

    assert result.returncode == 0, f"CLI failed:\n{result.stderr}"
    output = json.loads(result.stdout)
    assert len(output) == 1
    assert output[0]["patient"] == "DOE, JANE"
    assert len(output[0]["dr_files"]) == 2
    assert output[0]["verdict"] in {"VALID", "INVALID", "UNKNOWN_AGE"}

    xlsx_path = _REPO_ROOT / "output" / "dr_validity_results.xlsx"
    wb = openpyxl.load_workbook(xlsx_path)
    assert wb.sheetnames == [
        "All", "Valid", "Needs Review", "Groups", "Categories", "All Categories",
    ]

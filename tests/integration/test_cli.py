"""Integration tests — require real LiteParse npm CLI and a real PDF.

Run with: pytest -m integration
Skipped by default.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

DATA_DIR = Path(__file__).parents[2] / "data"
SAMPLE_PDF = DATA_DIR / "dr_progress_notes.pdf"


@pytest.mark.integration
def test_cli_processes_real_pdf() -> None:
    assert SAMPLE_PDF.exists(), f"Sample PDF not found at {SAMPLE_PDF}"

    result = subprocess.run(
        [sys.executable, "-m", "cli", "--input", str(SAMPLE_PDF)],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"CLI failed:\n{result.stderr}"
    output = json.loads(result.stdout)
    assert len(output) == 1
    assert output[0]["success"] is True
    assert output[0]["category"] in {"NEEDS_VISIT", "NO_VISIT_NEEDED"}
    assert output[0]["task_results"]["doctor_visit_needed"] in {"NEEDS_VISIT", "NO_VISIT_NEEDED"}
    assert output[0]["meta"]["patient_name"] is not None


@pytest.mark.integration
def test_cli_processes_real_folder() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "cli", "--input", str(DATA_DIR)],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"CLI failed:\n{result.stderr}"
    output = json.loads(result.stdout)
    assert len(output) >= 1

from dataclasses import dataclass
from pathlib import Path

from classifier.models import ClinicalVerdict
from config.settings import settings

_BASE = settings.base_path


@dataclass(frozen=True)
class PairExample:
    """One labeled dr+nurse example with ground-truth overall verdict.

    Each side is a list of paths; when it holds more than one file, their
    extracted text is appended (same behavior as the pair pipeline).
    """

    dr_paths: list[Path]
    nurse_paths: list[Path]
    expected: ClinicalVerdict
    description: str = ""


# ---------------------------------------------------------------------------
# Add labeled examples here. Paths must point to actual local files.
# ---------------------------------------------------------------------------
EXAMPLES: list[PairExample] = [
    PairExample(
        dr_paths=[_BASE / "cases/Active/ABAYEV, JONAH/dr_note.pdf"],
        nurse_paths=[_BASE / "cases/Active/ABAYEV, JONAH/nurse_note.pdf"],
        expected=ClinicalVerdict.MATCH,
        description="Same patient, same visit date",
    ),
    # Multiple notes per side — texts are appended:
    # PairExample(
    #     dr_paths=[_BASE / "cases/case2/dr_note.pdf", _BASE / "cases/case2/dr_note_2.pdf"],
    #     nurse_paths=[_BASE / "cases/case2/nurse_note.pdf"],
    #     expected=ClinicalVerdict.MISMATCH,
    #     description="Different patients, swapped files",
    # ),
]

from dataclasses import dataclass
from pathlib import Path

from classifier.models import ClinicalVerdict


@dataclass(frozen=True)
class PairExample:
    """One labeled dr+nurse file pair with ground-truth overall verdict."""

    dr_path: Path
    nurse_path: Path
    expected: ClinicalVerdict
    description: str = ""


# ---------------------------------------------------------------------------
# Add labeled examples here. Paths must point to actual local files.
# ---------------------------------------------------------------------------
EXAMPLES: list[PairExample] = [
    # PairExample(
    #     dr_path=Path("./classifier_data/case1/dr_note.pdf"),
    #     nurse_path=Path("./classifier_data/case1/nurse_note.pdf"),
    #     expected=ClinicalVerdict.MATCH,
    #     description="Same patient, same visit date",
    # ),
    # PairExample(
    #     dr_path=Path("./classifier_data/case2/dr_note.pdf"),
    #     nurse_path=Path("./classifier_data/case2/nurse_note.pdf"),
    #     expected=ClinicalVerdict.MISMATCH,
    #     description="Different patients, swapped files",
    # ),
]

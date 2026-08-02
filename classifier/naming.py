"""Note filename conventions: an optional category prefix plus a role prefix.

A note filename looks like ``[<category>_]<role>_<rest><ext>``, e.g. ``dr_note.pdf``,
``nurse_visit.txt``, ``hospital_nurse_visit.pdf``, ``peds_dr_progress_note.pdf``.
The category is optional; notes without one have ``category=None``, which is treated
as its own category for pairing purposes.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from config.settings import settings

logger = logging.getLogger(__name__)

NoteRole = Literal["dr", "nurse"]


@dataclass(frozen=True)
class NoteName:
    """A parsed note filename: role prefix plus optional category prefix."""

    role: NoteRole
    category: str | None = None


def _match_role(stem: str) -> NoteRole | None:
    """Return the role whose prefix stem starts with, or None."""
    for role, prefix in settings.note_role_prefixes.items():
        if stem.startswith(prefix.lower()):
            return cast(NoteRole, role)
    return None


def parse_note_name(name: str) -> NoteName | None:
    """Parse a filename into category + role. Returns None if it is not a note.

    A category prefix without a following role prefix (e.g. ``peds_summary.pdf``)
    is not a note.
    """
    lowered = name.lower()
    for category in settings.note_categories:
        prefix = f"{category.lower()}_"
        if lowered.startswith(prefix):
            role = _match_role(lowered[len(prefix):])
            return NoteName(role=role, category=category) if role else None
    role = _match_role(lowered)
    return NoteName(role=role) if role else None


def category_order() -> list[str | None]:
    """Categories in deterministic pairing order; uncategorized notes come first."""
    return [None, *settings.note_categories]


def is_supported_file(path: Path) -> bool:
    """Return True if path is a file with a supported extension."""
    return path.is_file() and path.suffix.lower() in settings.supported_extensions


def parse_note(path: Path, role: NoteRole | None = None) -> NoteName | None:
    """Parse a path as a supported note, optionally restricted to one role."""
    if not is_supported_file(path):
        return None
    parsed = parse_note_name(path.name)
    if parsed is None or (role is not None and parsed.role != role):
        return None
    return parsed


def label(category: str | None) -> str:
    """Human-readable category label used in log messages."""
    return category or "uncategorized"

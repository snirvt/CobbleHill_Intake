"""Folder-level checks: report leaf folders a pipeline cannot process, instead of skipping them."""

from pathlib import Path

from classifier.models import ErrorCode, ErrorStage, RunError
from classifier.naming import NoteRole, in_results_folder, parse_note


def leaf_folders(root: Path) -> list[Path]:
    """Every folder under root (root included) that has no subfolders, sorted.

    Results folders (where ``--upload-results`` puts reports) and their contents are
    ignored, so a patient folder holding only a results subfolder is still a leaf.
    """
    folders = [
        root,
        *sorted(p for p in root.rglob("*") if p.is_dir() and not in_results_folder(p, root)),
    ]
    return [
        f
        for f in folders
        if not any(c.is_dir() and not in_results_folder(c, root) for c in f.iterdir())
    ]


class RequiredNotesCheck:
    """A leaf folder must hold at least one note of every required role.

    Each classification sets its own roles: dr-validity needs ``("dr",)``; a pipeline
    pairing dr and nurse notes would use ``("dr", "nurse")``.
    """

    def __init__(self, roles: tuple[NoteRole, ...]) -> None:
        self._roles = roles

    def check(self, folder: Path) -> RunError | None:
        """Return an error when folder is empty or lacks a required note role, else None."""
        files = [f for f in folder.iterdir() if f.is_file()]
        if not files:
            return RunError(
                stage=ErrorStage.FOLDER,
                code=ErrorCode.EMPTY_FOLDER,
                message="Folder has no files",
                folder=folder,
            )
        missing = [role for role in self._roles if not any(parse_note(f, role=role) for f in files)]
        if missing:
            return RunError(
                stage=ErrorStage.FOLDER,
                code=ErrorCode.MISSING_REQUIRED_NOTES,
                message=f"Folder has no {' or '.join(missing)} notes",
                folder=folder,
            )
        return None

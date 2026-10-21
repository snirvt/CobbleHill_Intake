"""Which code the running server was started from (shown by GET /version)."""

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

UNKNOWN_COMMIT = "unknown"


def git_commit(repo: Path) -> str:
    """Short hash of the repo's HEAD commit, or "unknown" if git can't tell."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("Could not read git commit in %s: %s", repo, exc)
        return UNKNOWN_COMMIT
    return out.stdout.strip() or UNKNOWN_COMMIT

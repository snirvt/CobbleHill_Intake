import subprocess
from pathlib import Path
from unittest.mock import patch

from classifier.version import UNKNOWN_COMMIT, git_commit


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()


def test_git_commit_returns_short_head_hash(tmp_path: Path) -> None:
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q",
         "--allow-empty", "-m", "init")

    assert git_commit(tmp_path) == _git(tmp_path, "rev-parse", "--short", "HEAD")


def test_git_commit_outside_a_repo_is_unknown(tmp_path: Path) -> None:
    with patch.dict("os.environ", {"GIT_CEILING_DIRECTORIES": str(tmp_path.parent)}):
        assert git_commit(tmp_path) == UNKNOWN_COMMIT


def test_git_commit_without_git_installed_is_unknown(tmp_path: Path) -> None:
    with patch("classifier.version.subprocess.run", side_effect=OSError("no git")):
        assert git_commit(tmp_path) == UNKNOWN_COMMIT

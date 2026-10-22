"""Tests for machine-profile auto-detection and backend selection in config.settings."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from config import settings as settings_module
from config.settings import Settings, is_dgx_spark


@pytest.fixture
def on_machine(monkeypatch: pytest.MonkeyPatch):
    """Return a setter that fakes platform.machine() and clears related env overrides."""
    for var in ("COBBLEHILL_OCR_GPU", "COBBLEHILL_LLM_BACKEND", "COBBLEHILL_OLLAMA_MODEL"):
        monkeypatch.delenv(var, raising=False)

    def _set(arch: str) -> None:
        monkeypatch.setattr(settings_module.platform, "machine", lambda: arch)

    return _set


def test_is_dgx_spark_true_on_aarch64(on_machine) -> None:
    on_machine("aarch64")
    assert is_dgx_spark() is True


@pytest.mark.parametrize("arch", ["x86_64", "arm64", "AMD64"])
def test_is_dgx_spark_false_on_other_architectures(on_machine, arch: str) -> None:
    on_machine(arch)
    assert is_dgx_spark() is False


def test_settings_enables_ocr_gpu_on_spark(on_machine) -> None:
    on_machine("aarch64")
    assert Settings(_env_file=None).ocr_gpu is True


def test_settings_disables_ocr_gpu_on_laptop(on_machine) -> None:
    on_machine("x86_64")
    assert Settings(_env_file=None).ocr_gpu is False


def test_ocr_gpu_kwarg_overrides_auto_detection(on_machine) -> None:
    on_machine("aarch64")
    assert Settings(_env_file=None, ocr_gpu=False).ocr_gpu is False


@pytest.mark.parametrize("arch", ["aarch64", "x86_64"])
def test_llm_backend_defaults_to_ollama_on_every_machine(on_machine, arch: str) -> None:
    on_machine(arch)
    s = Settings(_env_file=None)
    assert s.llm_backend == "ollama"
    assert s.ollama_model == "gemma3:1b"


def test_llm_backend_env_selects_vllm(on_machine, monkeypatch: pytest.MonkeyPatch) -> None:
    on_machine("aarch64")
    monkeypatch.setenv("COBBLEHILL_LLM_BACKEND", "vllm")
    s = Settings(_env_file=None)
    assert s.llm_backend == "vllm"
    assert s.vllm_base_url == "http://localhost:8001"
    assert s.vllm_model == "nemotron-3.5-lightning"


def test_llm_backend_rejects_unknown_value(on_machine) -> None:
    on_machine("x86_64")
    with pytest.raises(ValidationError):
        Settings(_env_file=None, llm_backend="openai")


def test_task_thinking_defaults_to_off_for_every_task(on_machine) -> None:
    on_machine("x86_64")
    s = Settings(_env_file=None)
    assert s.task_thinking == {
        "age": "off", "diagnosis": "off", "care": "off", "main_diagnosis": "off",
        "clinical_match": "off",
    }
    assert set(s.vllm_thinking_levels) == {"off", "low", "high"}


def test_task_thinking_rejects_unknown_level(on_machine) -> None:
    on_machine("x86_64")
    with pytest.raises(ValidationError, match="unknown levels"):
        Settings(_env_file=None, task_thinking={"age": "medium"})


def test_task_thinking_env_override(on_machine, monkeypatch: pytest.MonkeyPatch) -> None:
    on_machine("x86_64")
    monkeypatch.setenv("COBBLEHILL_TASK_THINKING", '{"diagnosis": "low", "age": "high"}')
    s = Settings(_env_file=None)
    assert s.task_thinking == {"diagnosis": "low", "age": "high"}


def test_main_diagnosis_max_words_default_keeps_multi_word_labels() -> None:
    """Default is long enough that labels like "Failure To Thrive" are not cut mid-phrase."""
    assert Settings().main_diagnosis_max_words == 4


def _bin_dir(root, name: str, *executables: str) -> str:
    """Create a fake node bin dir under root holding the given executables."""
    d = root / name
    d.mkdir()
    for exe in executables:
        (d / exe).touch()
    return str(d)


def test_node_bin_path_skips_dir_without_liteparse(on_machine, tmp_path) -> None:
    on_machine("x86_64")
    node_only = _bin_dir(tmp_path, "system", "node")
    with_cli = _bin_dir(tmp_path, "nvm", "node", "liteparse")
    s = Settings(_env_file=None, node_bin_paths=[node_only, with_cli])
    assert s.node_bin_path == with_cli


def test_node_bin_path_raises_when_no_dir_has_liteparse(on_machine, tmp_path) -> None:
    on_machine("x86_64")
    node_only = _bin_dir(tmp_path, "system", "node")
    s = Settings(_env_file=None, node_bin_paths=[node_only])
    with pytest.raises(ValueError, match="liteparse"):
        _ = s.node_bin_path


def _nvm_version(home, version: str, *executables: str):
    """Create ~/.nvm/versions/node/<version>/bin under a fake home."""
    d = home / ".nvm" / "versions" / "node" / version / "bin"
    d.mkdir(parents=True)
    for exe in executables:
        (d / exe).touch()
    return d


def test_nvm_bin_dirs_returns_newest_version_first(tmp_path) -> None:
    for v in ("v9.1.0", "v22.22.3", "v22.22.10", "v18.19.1"):
        _nvm_version(tmp_path, v)
    names = [Path(p).parent.name for p in settings_module._nvm_bin_dirs(tmp_path)]
    assert names == ["v22.22.10", "v22.22.3", "v18.19.1", "v9.1.0"]


def test_nvm_bin_dirs_skips_non_version_dirs(tmp_path) -> None:
    _nvm_version(tmp_path, "v22.22.3")
    (tmp_path / ".nvm" / "versions" / "node" / "system").mkdir()
    assert len(settings_module._nvm_bin_dirs(tmp_path)) == 1


def test_nvm_bin_dirs_empty_without_nvm(tmp_path) -> None:
    assert settings_module._nvm_bin_dirs(tmp_path) == []


def test_default_node_bin_paths_puts_nvm_before_system(monkeypatch, tmp_path) -> None:
    nvm_bin = _nvm_version(tmp_path, "v22.22.3", "node", "liteparse")
    monkeypatch.setattr(settings_module.Path, "home", lambda: tmp_path)
    paths = settings_module._default_node_bin_paths()
    assert paths[0] == str(nvm_bin)
    assert paths.index("/usr/bin") > 0


def test_node_bin_path_picks_upgraded_nvm_over_system_node(
    on_machine, monkeypatch, tmp_path
) -> None:
    on_machine("x86_64")
    monkeypatch.delenv("COBBLEHILL_NODE_BIN_PATHS", raising=False)
    nvm_bin = _nvm_version(tmp_path, "v22.22.3", "node", "liteparse")
    monkeypatch.setattr(settings_module.Path, "home", lambda: tmp_path)
    assert Settings(_env_file=None).node_bin_path == str(nvm_bin)

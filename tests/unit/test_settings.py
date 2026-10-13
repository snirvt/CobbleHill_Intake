"""Tests for machine-profile auto-detection and backend selection in config.settings."""

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

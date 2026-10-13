"""Tests for machine-profile auto-detection in config.settings."""

import pytest

from config import settings as settings_module
from config.settings import Settings, is_dgx_spark


@pytest.fixture
def on_machine(monkeypatch: pytest.MonkeyPatch):
    """Return a setter that fakes platform.machine() and clears profile env overrides."""
    monkeypatch.delenv("COBBLEHILL_OLLAMA_MODEL", raising=False)
    monkeypatch.delenv("COBBLEHILL_OCR_GPU", raising=False)

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


def test_settings_picks_spark_model_and_gpu_on_aarch64(on_machine) -> None:
    on_machine("aarch64")
    s = Settings(_env_file=None)
    assert s.ollama_model == s.ollama_model_spark == "medgemma:27b"
    assert s.ocr_gpu is True


def test_settings_picks_laptop_model_and_cpu_ocr_on_x86_64(on_machine) -> None:
    on_machine("x86_64")
    s = Settings(_env_file=None)
    assert s.ollama_model == s.ollama_model_laptop == "gemma3:1b"
    assert s.ocr_gpu is False


def test_settings_uses_overridden_profile_model(on_machine) -> None:
    on_machine("aarch64")
    s = Settings(_env_file=None, ollama_model_spark="custom:7b")
    assert s.ollama_model == "custom:7b"


def test_ollama_model_env_overrides_auto_detection(
    on_machine, monkeypatch: pytest.MonkeyPatch
) -> None:
    on_machine("aarch64")
    monkeypatch.setenv("COBBLEHILL_OLLAMA_MODEL", "gemma3:1b")
    s = Settings(_env_file=None)
    assert s.ollama_model == "gemma3:1b"


def test_ocr_gpu_kwarg_overrides_auto_detection(on_machine) -> None:
    on_machine("aarch64")
    s = Settings(_env_file=None, ocr_gpu=False)
    assert s.ocr_gpu is False

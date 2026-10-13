"""Tests for LLM backend wiring in classifier.main."""

import asyncio

import pytest
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from classifier import main
from classifier.providers.ollama import SemaphoreChatModel


def test_make_llm_builds_ollama_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main.settings, "llm_backend", "ollama")
    llm = main._make_llm()
    assert isinstance(llm, SemaphoreChatModel)
    assert isinstance(llm.inner, ChatOllama)


def test_make_llm_ollama_ignores_thinking_level(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main.settings, "llm_backend", "ollama")
    llm = main._make_llm(thinking="off")
    assert isinstance(llm, SemaphoreChatModel)
    assert isinstance(llm.inner, ChatOllama)


def test_make_llm_builds_vllm_when_selected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main.settings, "llm_backend", "vllm")
    llm = main._make_llm()
    assert isinstance(llm, SemaphoreChatModel)
    assert isinstance(llm.inner, ChatOpenAI)
    assert llm.inner.model_name == main.settings.vllm_model


@pytest.mark.parametrize("level", ["off", "low", "high"])
def test_make_llm_vllm_sends_thinking_level_params(
    monkeypatch: pytest.MonkeyPatch, level: str
) -> None:
    monkeypatch.setattr(main.settings, "llm_backend", "vllm")
    llm = main._make_llm(thinking=level)
    assert isinstance(llm, SemaphoreChatModel)
    assert isinstance(llm.inner, ChatOpenAI)
    assert llm.inner.extra_body == main.settings.vllm_thinking_levels[level]


def test_make_llm_uses_given_semaphore(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main.settings, "llm_backend", "vllm")
    sem = asyncio.Semaphore(2)
    llm = main._make_llm(sem, "off")
    assert isinstance(llm, SemaphoreChatModel)
    assert llm.semaphore is sem


def test_dr_validity_pipeline_gives_each_task_its_level_and_one_semaphore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(main.settings, "llm_backend", "vllm")
    monkeypatch.setattr(
        main.settings,
        "task_thinking",
        {"age": "off", "diagnosis": "low", "care": "high", "main_diagnosis": "off"},
    )
    monkeypatch.setattr(main, "_make_env", lambda: None)
    monkeypatch.setattr(main, "_make_router", lambda: None)
    built: list[tuple[asyncio.Semaphore, str]] = []

    def fake_builder(semaphore: asyncio.Semaphore, thinking: str) -> str:
        built.append((semaphore, thinking))
        return thinking

    monkeypatch.setitem(main._LLM_BUILDERS, "vllm", fake_builder)
    captured: dict[str, object] = {}
    for name in (
        "build_patient_age_extractor",
        "build_diagnosis_extractor",
        "build_treatment_request_extractor",
        "build_main_diagnosis_extractor",
    ):
        monkeypatch.setattr(main, name, lambda llm, _n=name: captured.setdefault(_n, llm))

    main.build_dr_validity_pipeline()

    assert captured == {
        "build_patient_age_extractor": "off",
        "build_diagnosis_extractor": "low",
        "build_treatment_request_extractor": "high",
        "build_main_diagnosis_extractor": "off",
    }
    assert len({id(sem) for sem, _ in built}) == 1

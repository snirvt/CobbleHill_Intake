"""Tests for LLM backend wiring in classifier.main."""

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


def test_make_llm_builds_vllm_when_selected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main.settings, "llm_backend", "vllm")
    llm = main._make_llm()
    assert isinstance(llm, SemaphoreChatModel)
    assert isinstance(llm.inner, ChatOpenAI)
    assert llm.inner.model_name == main.settings.vllm_model

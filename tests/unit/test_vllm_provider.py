"""Tests for the vLLM provider and its on-prem guard."""

import asyncio

import pytest
from langchain_openai import ChatOpenAI

from classifier.providers.ollama import SemaphoreChatModel
from classifier.providers.vllm import assert_on_prem, create_vllm_chat_model


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:8001",
        "http://127.0.0.1:8001",
        "http://10.0.0.5:8001",
        "http://192.168.1.20:8001",
        "http://spark.local:8001",
        "http://[::1]:8001",
    ],
)
def test_assert_on_prem_accepts_local_and_private_hosts(url: str) -> None:
    assert_on_prem(url)


@pytest.mark.parametrize(
    "url",
    ["http://8.8.8.8:8001", "https://api.openai.com", "http://example.com:8001", ""],
)
def test_assert_on_prem_rejects_public_or_unknown_hosts(url: str) -> None:
    with pytest.raises(ValueError):
        assert_on_prem(url)


def test_create_vllm_chat_model_without_semaphore_returns_chat_openai() -> None:
    model = create_vllm_chat_model(
        base_url="http://localhost:8001/",
        model="nemotron-3.5-lightning",
        temperature=0.6,
        top_p=0.95,
    )
    assert isinstance(model, ChatOpenAI)
    assert model.openai_api_base == "http://localhost:8001/v1"
    assert model.model_name == "nemotron-3.5-lightning"
    assert model.temperature == 0.6
    assert model.top_p == 0.95


def test_create_vllm_chat_model_with_semaphore_returns_semaphore_wrapper() -> None:
    sem = asyncio.Semaphore(2)
    model = create_vllm_chat_model(
        base_url="http://localhost:8001",
        model="nemotron-3.5-lightning",
        temperature=0.6,
        top_p=0.95,
        semaphore=sem,
    )
    assert isinstance(model, SemaphoreChatModel)
    assert isinstance(model.inner, ChatOpenAI)
    assert model.semaphore is sem


def test_create_vllm_chat_model_refuses_public_base_url() -> None:
    with pytest.raises(ValueError):
        create_vllm_chat_model(
            base_url="https://api.openai.com",
            model="gpt",
            temperature=0.6,
            top_p=0.95,
        )


def test_create_vllm_chat_model_passes_extra_body() -> None:
    body = {"chat_template_kwargs": {"enable_thinking": False}}
    model = create_vllm_chat_model(
        base_url="http://localhost:8001",
        model="nemotron-3.5-lightning",
        temperature=0.6,
        top_p=0.95,
        extra_body=body,
    )
    assert isinstance(model, ChatOpenAI)
    assert model.extra_body == body

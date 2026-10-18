import asyncio
import logging

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import HumanMessage
from langchain_core.outputs import ChatResult

from classifier.providers.ollama import SemaphoreChatModel, create_ollama_chat_model


async def test_semaphore_chat_model_returns_inner_response() -> None:
    inner = FakeListChatModel(responses=["hello"])
    model = SemaphoreChatModel(inner=inner, semaphore=asyncio.Semaphore(5))
    result = await model.ainvoke([HumanMessage(content="hi")])
    assert result.content == "hello"


async def test_semaphore_chat_model_limits_concurrency() -> None:
    """Semaphore(1) serialises concurrent calls."""
    call_order: list[str] = []

    class TrackingModel(FakeListChatModel):
        responses: list[str] = ["ok", "ok"]

        async def _agenerate(self, messages, **kwargs):  # type: ignore[override]
            call_order.append("start")
            await asyncio.sleep(0)
            call_order.append("end")
            return await super()._agenerate(messages, **kwargs)

    inner = TrackingModel(responses=["ok", "ok"])
    model = SemaphoreChatModel(inner=inner, semaphore=asyncio.Semaphore(1))

    msg = [HumanMessage(content="x")]
    await asyncio.gather(model.ainvoke(msg), model.ainvoke(msg))

    assert call_order == ["start", "end", "start", "end"]


def test_create_ollama_chat_model_without_semaphore_returns_chat_ollama() -> None:
    from langchain_ollama import ChatOllama

    model = create_ollama_chat_model(url="http://localhost:11434", model="gemma3:1b")
    assert isinstance(model, ChatOllama)


def test_create_ollama_chat_model_with_semaphore_returns_semaphore_wrapper() -> None:
    sem = asyncio.Semaphore(3)
    model = create_ollama_chat_model(url="http://localhost:11434", model="gemma3:1b", semaphore=sem)
    assert isinstance(model, SemaphoreChatModel)
    assert model.semaphore is sem


async def test_semaphore_chat_model_records_timing_and_usage_under_active_timings() -> None:
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    from classifier.timing import PatientTimings, current_timings, step

    class UsageModel(FakeListChatModel):
        async def _agenerate(self, messages, **kwargs):  # type: ignore[override]
            msg = AIMessage(
                content='{"a": 1}',
                usage_metadata={"input_tokens": 120, "output_tokens": 30, "total_tokens": 150},
            )
            return ChatResult(generations=[ChatGeneration(message=msg)])

    model = SemaphoreChatModel(inner=UsageModel(responses=["x"]), semaphore=asyncio.Semaphore(1))
    timings = PatientTimings(patient="P")
    token = current_timings.set(timings)
    try:
        await step("diagnosis", model.ainvoke([HumanMessage(content="hi")]))
    finally:
        current_timings.reset(token)

    [call] = timings.llm_calls
    assert call.step == "diagnosis"
    assert call.prompt_tokens == 120
    assert call.completion_tokens == 30
    assert call.answer_chars == len('{"a": 1}')
    assert call.wait_s >= 0 and call.call_s >= 0


async def test_semaphore_chat_model_records_nothing_without_active_timings() -> None:
    from classifier.timing import current_timings

    model = SemaphoreChatModel(
        inner=FakeListChatModel(responses=["ok"]), semaphore=asyncio.Semaphore(1)
    )
    await model.ainvoke([HumanMessage(content="hi")])
    assert current_timings.get() is None


def test_create_ollama_chat_model_caps_tokens_and_time() -> None:
    from langchain_ollama import ChatOllama

    model = create_ollama_chat_model(
        url="http://localhost:11434", model="gemma3:1b", max_tokens=8192, timeout_s=300
    )
    assert isinstance(model, ChatOllama)
    assert model.num_predict == 8192
    assert model.client_kwargs == {"timeout": 300}


def _result_with(**info: str) -> ChatResult:
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration

    return ChatResult(
        generations=[ChatGeneration(message=AIMessage(content="x"), generation_info=info)]
    )


async def test_semaphore_chat_model_warns_when_output_hits_token_limit(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class TruncatedModel(FakeListChatModel):
        async def _agenerate(self, messages, **kwargs):  # type: ignore[override]
            return _result_with(finish_reason="length")

    model = SemaphoreChatModel(
        inner=TruncatedModel(responses=["x"]), semaphore=asyncio.Semaphore(1)
    )

    with caplog.at_level(logging.WARNING, logger="classifier.providers.ollama"):
        await model.ainvoke([HumanMessage(content="hi")])

    assert any("max-token limit" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize(
    ("info", "expected"),
    [
        ({"finish_reason": "length"}, True),
        ({"done_reason": "length"}, True),
        ({"finish_reason": "stop"}, False),
        ({}, False),
    ],
)
def test_hit_token_limit_reads_vllm_and_ollama_reasons(
    info: dict[str, str], expected: bool
) -> None:
    from classifier.providers.ollama import _hit_token_limit

    assert _hit_token_limit(_result_with(**info)) is expected

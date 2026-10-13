import asyncio

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import HumanMessage

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

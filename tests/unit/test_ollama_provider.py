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

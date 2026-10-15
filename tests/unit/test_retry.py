import pytest
from langchain_core.runnables import RunnableLambda

from classifier.classifiers.retry import with_llm_retry
from config.settings import settings


def _flaky(failures: int) -> tuple[RunnableLambda[str, str], list[str]]:
    calls: list[str] = []

    def run(value: str) -> str:
        calls.append(value)
        if len(calls) <= failures:
            raise ConnectionError("ollama down")
        return value.upper()

    return RunnableLambda(run), calls


async def test_with_llm_retry_recovers_from_transient_failures() -> None:
    chain, calls = _flaky(failures=settings.llm_max_attempts - 1)

    assert await with_llm_retry(chain).ainvoke("ok") == "OK"
    assert len(calls) == settings.llm_max_attempts


async def test_with_llm_retry_gives_up_after_max_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "llm_max_attempts", 2)
    chain, calls = _flaky(failures=5)

    with pytest.raises(ConnectionError, match="ollama down"):
        await with_llm_retry(chain).ainvoke("ok")
    assert len(calls) == 2

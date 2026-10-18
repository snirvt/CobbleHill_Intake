import asyncio
import logging
import time
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from langchain_ollama import ChatOllama

from classifier.timing import record_llm_call

logger = logging.getLogger(__name__)


class SemaphoreChatModel(BaseChatModel):
    """Wraps any BaseChatModel with an asyncio.Semaphore to cap concurrent LLM calls."""

    inner: BaseChatModel
    semaphore: Any  # asyncio.Semaphore — not serialisable by Pydantic, typed as Any

    @property
    def _llm_type(self) -> str:
        return f"semaphore-{self.inner._llm_type}"

    def _generate(self, messages: list[BaseMessage], **kwargs: Any) -> ChatResult:
        raise NotImplementedError("Use async path only")

    async def _agenerate(self, messages: list[BaseMessage], **kwargs: Any) -> ChatResult:
        queued = time.perf_counter()
        async with self.semaphore:
            started = time.perf_counter()
            logger.debug("LLM call start messages=%d", len(messages))
            result = await self.inner._agenerate(messages, **kwargs)
            finished = time.perf_counter()
            logger.debug("LLM call done in %.2fs", finished - started)
        if _hit_token_limit(result):
            logger.warning(
                "LLM output hit the max-token limit after %.2fs; answer is truncated",
                finished - started,
            )
        _record_timing(result, wait_s=started - queued, call_s=finished - started)
        return result


def _hit_token_limit(result: ChatResult) -> bool:
    """True if generation stopped on the token cap (vLLM finish_reason / ollama done_reason)."""
    for gen in result.generations:
        info = {**(gen.generation_info or {}), **gen.message.response_metadata}
        if "length" in (info.get("finish_reason"), info.get("done_reason")):
            return True
    return False


def _record_timing(result: ChatResult, wait_s: float, call_s: float) -> None:
    """Report the call's timing and token usage to the active patient timings, if any."""
    message = result.generations[0].message if result.generations else None
    usage = getattr(message, "usage_metadata", None) or {}
    record_llm_call(
        wait_s=wait_s,
        call_s=call_s,
        prompt_tokens=usage.get("input_tokens"),
        completion_tokens=usage.get("output_tokens"),
        answer_chars=len(message.text) if message is not None else 0,
    )


def create_ollama_chat_model(
    url: str,
    model: str,
    semaphore: asyncio.Semaphore | None = None,
    max_tokens: int | None = None,
    timeout_s: float | None = None,
) -> BaseChatModel:
    """Return a ChatOllama instance, optionally wrapped in a concurrency semaphore.

    ``max_tokens`` caps generation per call (ollama ``num_predict``); ``timeout_s`` caps
    each request's wall-clock time.
    """
    inner = ChatOllama(
        base_url=url.rstrip("/"),
        model=model,
        num_predict=max_tokens,
        client_kwargs={"timeout": timeout_s} if timeout_s is not None else {},
    )
    if semaphore is None:
        return inner
    return SemaphoreChatModel(inner=inner, semaphore=semaphore)

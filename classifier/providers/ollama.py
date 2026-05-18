import asyncio
import logging
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from langchain_ollama import ChatOllama

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
        async with self.semaphore:
            logger.debug("LLM call start messages=%d", len(messages))
            result = await self.inner._agenerate(messages, **kwargs)
            logger.debug("LLM call done")
            return result


def create_ollama_chat_model(
    url: str,
    model: str,
    semaphore: asyncio.Semaphore | None = None,
) -> BaseChatModel:
    """Return a ChatOllama instance, optionally wrapped in a concurrency semaphore."""
    inner = ChatOllama(base_url=url.rstrip("/"), model=model)
    if semaphore is None:
        return inner
    return SemaphoreChatModel(inner=inner, semaphore=semaphore)

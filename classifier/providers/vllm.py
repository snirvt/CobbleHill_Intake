import asyncio
import ipaddress
import logging
from typing import Any
from urllib.parse import urlparse

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from classifier.providers.ollama import SemaphoreChatModel
from classifier.providers.structured import vllm_schema_kwargs

logger = logging.getLogger(__name__)


def assert_on_prem(base_url: str) -> None:
    """Refuse a public base_url so prompts (incl. patient data) never leave the network.

    Only ``localhost``, ``*.local`` and loopback / private / link-local IPs are allowed.
    """
    host = urlparse(base_url).hostname or ""
    if host == "localhost" or host.endswith(".local"):
        return
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        raise ValueError(
            f"vLLM host {host!r} is not on-prem; refusing to send patient data off-box"
        ) from None
    if not (ip.is_loopback or ip.is_private or ip.is_link_local):
        raise ValueError(
            f"vLLM base_url {base_url!r} is a public address; refusing to send patient data"
        )


def create_vllm_chat_model(
    base_url: str,
    model: str,
    temperature: float,
    top_p: float,
    semaphore: asyncio.Semaphore | None = None,
    extra_body: dict[str, Any] | None = None,
    max_tokens: int | None = None,
    timeout_s: float | None = None,
) -> BaseChatModel:
    """Return a ChatOpenAI client for an on-prem vLLM server, optionally semaphore-wrapped.

    ``extra_body`` is merged into every request (e.g. vLLM ``chat_template_kwargs``).
    ``max_tokens`` caps generation per call; ``timeout_s`` caps its wall-clock time. The
    client's own retries are disabled: the extractors' ``with_llm_retry`` handles retrying.
    """
    assert_on_prem(base_url)
    inner = ChatOpenAI(
        model=model,
        base_url=base_url.rstrip("/") + "/v1",
        api_key=SecretStr("EMPTY"),  # vLLM ignores it; the client requires one
        temperature=temperature,
        top_p=top_p,
        extra_body=extra_body,
        max_completion_tokens=max_tokens,  # sent as max_completion_tokens; vLLM honours it
        timeout=timeout_s,
        max_retries=0,
    )
    if semaphore is None:
        return inner
    return SemaphoreChatModel(inner=inner, semaphore=semaphore, schema_kwargs=vllm_schema_kwargs)

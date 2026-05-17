import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from classifier.providers.ollama import OllamaProvider


def _mock_response(text: str) -> MagicMock:
    resp = MagicMock()
    resp.json.return_value = {"response": text}
    resp.raise_for_status = MagicMock()
    return resp


@patch("classifier.providers.ollama.httpx.AsyncClient")
async def test_ollama_provider_returns_response_text(mock_client_cls) -> None:
    mock_client = AsyncMock()
    mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=_mock_response("NEEDS_VISIT"))

    provider = OllamaProvider(url="http://localhost:11434", model="gemma3:1b")
    result = await provider.complete("some prompt")

    assert result == "NEEDS_VISIT"
    mock_client.post.assert_awaited_once()
    call_kwargs = mock_client.post.call_args
    assert call_kwargs[0][0] == "http://localhost:11434/api/generate"
    assert call_kwargs[1]["json"]["model"] == "gemma3:1b"
    assert call_kwargs[1]["json"]["prompt"] == "some prompt"
    assert call_kwargs[1]["json"]["stream"] is False


@patch("classifier.providers.ollama.httpx.AsyncClient")
async def test_ollama_provider_strips_trailing_slash(mock_client_cls) -> None:
    mock_client = AsyncMock()
    mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=_mock_response("ok"))

    provider = OllamaProvider(url="http://localhost:11434/", model="m")
    await provider.complete("p")

    url = mock_client.post.call_args[0][0]
    assert url == "http://localhost:11434/api/generate"


@patch("classifier.providers.ollama.httpx.AsyncClient")
async def test_ollama_provider_semaphore_limits_concurrency(mock_client_cls) -> None:
    """Semaphore with value=1 should serialize concurrent calls."""
    call_order: list[str] = []

    async def slow_post(*args, **kwargs):  # type: ignore[no-untyped-def]
        call_order.append("start")
        await asyncio.sleep(0)
        call_order.append("end")
        return _mock_response("ok")

    mock_client = AsyncMock()
    mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = slow_post

    sem = asyncio.Semaphore(1)
    provider = OllamaProvider(url="http://localhost:11434", model="m", semaphore=sem)

    await asyncio.gather(provider.complete("a"), provider.complete("b"))

    # With semaphore=1 calls serialize: start→end→start→end
    assert call_order == ["start", "end", "start", "end"]

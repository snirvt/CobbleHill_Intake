"""Constrained JSON output: ask the LLM server to only emit JSON matching a schema."""

from collections.abc import Callable
from typing import Any

from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable
from pydantic import BaseModel

# Maps an output model to the request kwargs that constrain generation to its schema.
SchemaKwargs = Callable[[type[BaseModel]], dict[str, Any]]


def ollama_schema_kwargs(model: type[BaseModel]) -> dict[str, Any]:
    """Ollama structured outputs: ``format`` takes the JSON schema itself."""
    return {"format": model.model_json_schema()}


def vllm_schema_kwargs(model: type[BaseModel]) -> dict[str, Any]:
    """vLLM (OpenAI-compatible) structured outputs via ``response_format: json_schema``."""
    return {
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": model.__name__, "schema": model.model_json_schema()},
        }
    }


def bind_output_schema(
    llm: BaseChatModel, model: type[BaseModel]
) -> Runnable[LanguageModelInput, BaseMessage]:
    """Bind ``model``'s JSON schema to ``llm`` if its provider supports constrained output.

    Providers opt in by setting a ``schema_kwargs`` callable on the chat model (see
    ``SemaphoreChatModel``); any other model (e.g. the test stub) is returned unchanged.
    """
    schema_kwargs: SchemaKwargs | None = getattr(llm, "schema_kwargs", None)
    if schema_kwargs is None:
        return llm
    return llm.bind(**schema_kwargs(model))

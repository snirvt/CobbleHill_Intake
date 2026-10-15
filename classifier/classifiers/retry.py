"""Shared retry policy for LLM extraction chains."""

from langchain_core.runnables import Runnable

from config.settings import settings


def with_llm_retry[In, Out](chain: Runnable[In, Out]) -> Runnable[In, Out]:
    """Retry the whole chain (call + parse) up to settings.llm_max_attempts times.

    Covers transport failures and malformed output alike; backoff starts at
    settings.llm_retry_wait_s and doubles, with up to that much jitter.
    """
    wait = settings.llm_retry_wait_s
    return chain.with_retry(
        stop_after_attempt=settings.llm_max_attempts,
        wait_exponential_jitter=wait > 0,
        exponential_jitter_params={"initial": wait, "jitter": wait},
    )

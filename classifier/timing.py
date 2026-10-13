"""Per-patient timing collection for the dr-validity run.

A pipeline sets ``current_timings`` for one patient; extraction and LLM-call timings are
appended to it from anywhere below (including concurrent ``asyncio.gather`` tasks, which
inherit a copy of the context). Without a ``current_timings`` nothing is recorded, so code
paths that never set it (CLI pipelines) are unaffected.
"""

import logging
from collections.abc import Awaitable
from contextvars import ContextVar
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Rough chars-per-token ratio, used only to estimate how much output was the JSON answer.
_CHARS_PER_TOKEN = 4


@dataclass
class ExtractTiming:
    """How long one file took to turn into text."""

    file: str
    extractor: str
    pages: int | None
    seconds: float


@dataclass
class LlmCallTiming:
    """One LLM call: time waiting for a concurrency slot, call time and token usage."""

    step: str
    wait_s: float
    call_s: float
    prompt_tokens: int | None
    completion_tokens: int | None
    answer_chars: int

    @property
    def thinking_tokens_estimate(self) -> int | None:
        """Output tokens not accounted for by the answer text (≈ reasoning tokens)."""
        if self.completion_tokens is None:
            return None
        return max(0, self.completion_tokens - self.answer_chars // _CHARS_PER_TOKEN)


@dataclass
class PatientTimings:
    """All timings recorded while processing one patient."""

    patient: str
    extracts: list[ExtractTiming] = field(default_factory=list)
    llm_calls: list[LlmCallTiming] = field(default_factory=list)
    extract_wall_s: float = 0.0
    classify_wall_s: float = 0.0
    total_s: float = 0.0

    def summary_lines(self) -> list[str]:
        """Human-readable log lines: one per file, one per LLM call, one total."""
        prefix = f"timing patient={self.patient}"
        lines = [
            f"{prefix} extract file={e.file} extractor={e.extractor} "
            f"pages={e.pages if e.pages is not None else '?'} {e.seconds:.2f}s"
            for e in self.extracts
        ]
        for c in self.llm_calls:
            rate = (
                f"{c.completion_tokens / c.call_s:.1f} out_tok/s"
                if c.completion_tokens is not None and c.call_s > 0
                else "? out_tok/s"
            )
            lines.append(
                f"{prefix} llm step={c.step} wait={c.wait_s:.2f}s call={c.call_s:.2f}s "
                f"prompt_tok={_fmt(c.prompt_tokens)} out_tok={_fmt(c.completion_tokens)} "
                f"(~thinking {_fmt(c.thinking_tokens_estimate)}) {rate}"
            )
        prompt_total = sum(c.prompt_tokens or 0 for c in self.llm_calls)
        out_total = sum(c.completion_tokens or 0 for c in self.llm_calls)
        lines.append(
            f"{prefix} TOTAL {self.total_s:.2f}s | extract {self.extract_wall_s:.2f}s | "
            f"llm wall {self.classify_wall_s:.2f}s | {len(self.llm_calls)} calls, "
            f"prompt_tok={prompt_total} out_tok={out_total}"
        )
        return lines

    def log(self) -> None:
        """Emit summary_lines at INFO."""
        for line in self.summary_lines():
            logger.info(line)


def _fmt(value: int | None) -> str:
    return "?" if value is None else str(value)


current_timings: ContextVar[PatientTimings | None] = ContextVar("current_timings", default=None)
current_step: ContextVar[str] = ContextVar("current_step", default="unknown")


def record_llm_call(
    wait_s: float,
    call_s: float,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    answer_chars: int,
) -> None:
    """Append an LLM call to the active patient's timings; no-op when none is active."""
    timings = current_timings.get()
    if timings is None:
        return
    timings.llm_calls.append(
        LlmCallTiming(
            step=current_step.get(),
            wait_s=wait_s,
            call_s=call_s,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            answer_chars=answer_chars,
        )
    )


async def step[T](name: str, awaitable: Awaitable[T]) -> T:
    """Await ``awaitable`` with LLM calls inside it attributed to step ``name``.

    Run it as its own task (e.g. an ``asyncio.gather`` argument) so concurrent steps keep
    separate names.
    """
    current_step.set(name)
    return await awaitable

"""Tests for per-patient timing collection."""

import asyncio
import contextvars
import logging

import pytest

from classifier.timing import (
    ExtractTiming,
    LlmCallTiming,
    PatientTimings,
    current_timings,
    record_llm_call,
    step,
)


def _run_in_context(coro_fn):
    """Run an async function in a fresh context so contextvar sets don't leak between tests."""
    return contextvars.copy_context().run(asyncio.run, coro_fn())


def test_record_llm_call_is_noop_without_active_timings() -> None:
    async def body() -> None:
        record_llm_call(wait_s=0.1, call_s=1.0, prompt_tokens=10, completion_tokens=5,
                        answer_chars=4)
        assert current_timings.get() is None

    _run_in_context(body)


def test_record_llm_call_appends_to_active_timings_with_step_name() -> None:
    async def body() -> PatientTimings:
        timings = PatientTimings(patient="P")
        current_timings.set(timings)

        async def call() -> None:
            record_llm_call(wait_s=0.5, call_s=2.0, prompt_tokens=100, completion_tokens=40,
                            answer_chars=40)

        await step("age", call())
        return timings

    timings = _run_in_context(body)
    assert timings.llm_calls == [
        LlmCallTiming(step="age", wait_s=0.5, call_s=2.0, prompt_tokens=100,
                      completion_tokens=40, answer_chars=40)
    ]


def test_step_keeps_names_separate_across_concurrent_gather() -> None:
    async def body() -> PatientTimings:
        timings = PatientTimings(patient="P")
        current_timings.set(timings)

        async def call(delay: float) -> None:
            await asyncio.sleep(delay)
            record_llm_call(wait_s=0, call_s=delay, prompt_tokens=1, completion_tokens=1,
                            answer_chars=0)

        await asyncio.gather(step("slow", call(0.02)), step("fast", call(0.0)))
        return timings

    timings = _run_in_context(body)
    by_step = {c.step: c.call_s for c in timings.llm_calls}
    assert by_step == {"slow": 0.02, "fast": 0.0}


def test_thinking_tokens_estimate_subtracts_answer_tokens() -> None:
    call = LlmCallTiming(step="s", wait_s=0, call_s=1, prompt_tokens=10,
                         completion_tokens=1000, answer_chars=400)
    assert call.thinking_tokens_estimate == 900


def test_thinking_tokens_estimate_unknown_without_usage() -> None:
    call = LlmCallTiming(step="s", wait_s=0, call_s=1, prompt_tokens=None,
                         completion_tokens=None, answer_chars=400)
    assert call.thinking_tokens_estimate is None


def test_summary_lines_list_files_calls_and_totals() -> None:
    timings = PatientTimings(
        patient="P1",
        extracts=[ExtractTiming(file="dr_a.pdf", extractor="PdfExtractor", pages=3, seconds=1.5)],
        llm_calls=[
            LlmCallTiming(step="age", wait_s=0.25, call_s=10.0, prompt_tokens=500,
                          completion_tokens=200, answer_chars=80),
            LlmCallTiming(step="care", wait_s=0.0, call_s=5.0, prompt_tokens=None,
                          completion_tokens=None, answer_chars=10),
        ],
        extract_wall_s=1.5,
        classify_wall_s=10.0,
        total_s=11.5,
    )

    lines = timings.summary_lines()

    assert lines[0] == (
        "timing patient=P1 extract file=dr_a.pdf extractor=PdfExtractor pages=3 1.50s"
    )
    assert lines[1] == (
        "timing patient=P1 llm step=age wait=0.25s call=10.00s prompt_tok=500 out_tok=200 "
        "(~thinking 180) 20.0 out_tok/s"
    )
    assert lines[2] == (
        "timing patient=P1 llm step=care wait=0.00s call=5.00s prompt_tok=? out_tok=? "
        "(~thinking ?) ? out_tok/s"
    )
    assert lines[3] == (
        "timing patient=P1 TOTAL 11.50s | extract 1.50s | llm wall 10.00s | 2 calls, "
        "prompt_tok=500 out_tok=200"
    )


def test_log_emits_summary_lines_at_info(caplog: pytest.LogCaptureFixture) -> None:
    timings = PatientTimings(patient="P1", total_s=1.0)
    with caplog.at_level(logging.INFO, logger="classifier.timing"):
        timings.log()
    assert [r.getMessage() for r in caplog.records] == timings.summary_lines()

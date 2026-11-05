"""Guard for free-text LLM reasoning fields against degenerate output.

A model stuck in a repetition loop can still emit valid JSON (constrained decoding keeps
the shape) while the reasoning string fills with repeated sentences or with the prompt's
own output instructions echoed back. ``ReasoningText`` is the string type every output
model uses for reasoning: its validator rejects such text, so the output parser raises
``OutputParserException``, ``with_llm_retry`` retries the call and, on the final failure,
the step reports an error instead of passing the text on as a real answer.
"""

import re
from collections import Counter
from typing import Annotated

from pydantic import AfterValidator

from config.settings import settings

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_NON_WORD = re.compile(r"[^\w]+")
_NGRAM = 4
# Sentences shorter than this (in words) are not counted as repeats ("Stable.").
_MIN_SENTENCE_WORDS = 4


def _normalize(text: str) -> str:
    """Comparison key: lower case, punctuation collapsed to single spaces."""
    return _NON_WORD.sub(" ", text).strip().casefold()


def _repeated_sentence(text: str) -> str | None:
    """The first sentence appearing at least ``reasoning_max_sentence_repeats`` times."""
    sentences = [_normalize(s) for s in _SENTENCE_SPLIT.split(text)]
    counts = Counter(s for s in sentences if len(s.split()) >= _MIN_SENTENCE_WORDS)
    for sentence, count in counts.most_common(1):
        if count >= settings.reasoning_max_sentence_repeats:
            return sentence
    return None


def _distinct_ngram_ratio(words: list[str]) -> float:
    """Share of distinct word 4-grams; 1.0 means no 4-gram repeats."""
    ngrams = [tuple(words[i : i + _NGRAM]) for i in range(len(words) - _NGRAM + 1)]
    return len(set(ngrams)) / len(ngrams) if ngrams else 1.0


def find_reasoning_problem(text: str | None) -> str | None:
    """Describe why ``text`` looks degenerate, or None when it looks like real reasoning."""
    if not text:
        return None
    if len(text) > settings.reasoning_max_chars:
        return f"reasoning too long ({len(text)} > {settings.reasoning_max_chars} chars)"
    lowered = text.casefold()
    leaked = [m for m in settings.reasoning_leak_markers if m.casefold() in lowered]
    if leaked:
        return f"reasoning echoes output instructions ({', '.join(leaked)})"
    repeated = _repeated_sentence(text)
    if repeated is not None:
        return f"reasoning repeats a sentence: {repeated!r}"
    words = _normalize(text).split()
    if len(words) >= settings.reasoning_ngram_min_words:
        ratio = _distinct_ngram_ratio(words)
        if ratio < settings.reasoning_min_distinct_ngram_ratio:
            return f"reasoning is mostly repetition ({ratio:.0%} distinct phrases)"
    return None


def _validate_reasoning(text: str) -> str:
    """Pydantic validator: raise ValueError for degenerate reasoning, else return it."""
    problem = find_reasoning_problem(text)
    if problem is not None:
        raise ValueError(f"degenerate LLM output: {problem}")
    return text


# Reasoning string for LLM output models, validated against loops / echoed instructions.
ReasoningText = Annotated[str, AfterValidator(_validate_reasoning)]

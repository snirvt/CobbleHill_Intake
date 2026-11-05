"""Bucketing step: put each already-extracted diagnosis into a DiagnosisCategory.

Runs separately from extraction so the bucketing prompt can be tuned on plain diagnosis
strings, without a note. ``CategorizedDiagnosisExtractor`` chains the two steps.
"""

import logging
import re

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from classifier.classifiers.categories import (
    CATEGORY_EXAMPLES,
    CATEGORY_LIST,
    CATEGORY_RULES,
)
from classifier.classifiers.diagnosis_extraction import DiagnosisExtractionClassifier
from classifier.classifiers.reasoning_guard import ReasoningText
from classifier.classifiers.retry import with_llm_retry
from classifier.models import (
    Diagnosis,
    DiagnosisCategorizationResult,
    DiagnosisCategory,
    DiagnosisExtractionResult,
    DocumentMetadata,
    unique_categories,
)
from classifier.protocols import DiagnosisCategorizer
from classifier.providers.structured import bind_output_schema
from config.settings import settings

logger = logging.getLogger(__name__)


class CategorizedItem(BaseModel):
    """One diagnosis echoed back with its bucket."""

    name: str
    # Required and before category: makes the model reason per item (e.g. weigh the age)
    # even with thinking off, instead of picking the bucket in one step.
    reason: ReasoningText
    category: DiagnosisCategory


class DiagnosisCategorizationOutput(BaseModel):
    """Structured LLM output for the categorization task."""

    # Declared first so the model reasons before picking buckets.
    reasoning: ReasoningText | None = None
    # Required: one item per input diagnosis, same order.
    categories: list[CategorizedItem]


_PROMPT_TEMPLATE = """\
You are a medical coding assistant. Put each diagnosis below into exactly one category \
from this list (copy the text exactly):
{categories}

Category rules:
{category_rules}

Examples (diagnosis, patient age → category):
{category_examples}

Return exactly one item per diagnosis, in the same order as the numbered list, echoing \
its name. For each item give a short reason first (one clause), naming the rule you \
applied and the patient age when that rule depends on age; then the category.

{format_instructions}

Check each diagnosis against the rules and examples above before choosing; the patient age \
label (infant / not an infant) applies to every diagnosis.

Patient age: {patient_age}

--- DIAGNOSES ---
{diagnoses}
--- END ---"""


def _join_reasons(output: DiagnosisCategorizationOutput) -> str | None:
    """Overall reasoning (if any) plus one ``name: reason`` line per item."""
    lines = [output.reasoning] if output.reasoning else []
    lines += [f"{item.name}: {item.reason}" for item in output.categories if item.reason]
    return "\n".join(lines) or None


def _format_diagnoses(diagnoses: list[Diagnosis]) -> str:
    """Numbered prompt lines: ``1. Underweight (R63.6)``; the ICD part only when present."""
    return "\n".join(
        f"{i}. {d.name} ({d.icd_code})" if d.icd_code else f"{i}. {d.name}"
        for i, d in enumerate(diagnoses, start=1)
    )


_ICD_SUFFIX = re.compile(r"\s*\([^()]*\)\s*$")
_NON_WORD = re.compile(r"[^\w]+")


def _normalize_name(name: str) -> str:
    """Comparison key for an echoed diagnosis name: no trailing ``(ICD)``, case or punctuation."""
    return _NON_WORD.sub(" ", _ICD_SUFFIX.sub("", name)).strip().casefold()


def count_mismatch_warning(expected: int, got: int, uncategorized: int, attempts: int) -> str:
    """Report text for a categorization still returning the wrong item count after retries."""
    return (
        f"Category count mismatch after {attempts} attempts: expected {expected}, got {got}; "
        f"{uncategorized} diagnoses left uncategorized"
    )


def _match_by_name(
    diagnoses: list[Diagnosis], items: list[CategorizedItem]
) -> list[CategorizedItem | None]:
    """Pair each diagnosis with the first unused item of the same normalized name.

    Used when the model returned a different number of items than diagnoses, so position
    can't be trusted. Duplicate names consume items in order; unmatched diagnoses get None
    and extra items are ignored.
    """
    pool: dict[str, list[CategorizedItem]] = {}
    for item in items:
        pool.setdefault(_normalize_name(item.name), []).append(item)
    matched: list[CategorizedItem | None] = []
    for diagnosis in diagnoses:
        candidates = pool.get(_normalize_name(diagnosis.name))
        matched.append(candidates.pop(0) if candidates else None)
    return matched


def _apply(diagnosis: Diagnosis, item: CategorizedItem | None) -> Diagnosis:
    """The diagnosis with the item's category and reason; unchanged when item is None."""
    if item is None:
        return diagnosis
    return diagnosis.model_copy(update={"category": item.category, "category_reason": item.reason})


class DiagnosisCategorizationClassifier:
    """Assigns a DiagnosisCategory to every diagnosis of a note in one LLM call."""

    def __init__(self, llm: BaseChatModel) -> None:
        parser: PydanticOutputParser[DiagnosisCategorizationOutput] = PydanticOutputParser(
            pydantic_object=DiagnosisCategorizationOutput
        )
        prompt = ChatPromptTemplate.from_template(_PROMPT_TEMPLATE).partial(
            format_instructions=parser.get_format_instructions(),
            categories=CATEGORY_LIST,
            category_rules=CATEGORY_RULES,
            category_examples=CATEGORY_EXAMPLES,
        )
        self._chain = with_llm_retry(
            prompt | bind_output_schema(llm, DiagnosisCategorizationOutput) | parser
        )

    async def categorize(
        self, diagnoses: list[Diagnosis], patient_age: str | None = None
    ) -> DiagnosisCategorizationResult:
        """Return the diagnoses with categories set, by position; error set on failure.

        A reply with the wrong number of items is retried, up to
        ``settings.category_mismatch_max_attempts`` calls in total; after the last one the
        items are matched by name and the result carries a warning instead of an error.

        ``patient_age`` (e.g. "3 months", from the patient-age step) decides child-vs-adult
        buckets; None is shown to the model as "unknown".
        """
        if not diagnoses:
            return DiagnosisCategorizationResult()
        attempts = max(1, settings.category_mismatch_max_attempts)
        for attempt in range(1, attempts + 1):
            try:
                output = await self._call_once(diagnoses, patient_age)
            except OutputParserException as exc:
                logger.error("Category parse failed: %s", exc)
                return DiagnosisCategorizationResult(
                    diagnoses=diagnoses, error=f"Category parse error: {exc}"
                )
            except Exception as exc:
                logger.error("Category LLM call failed: %s", exc)
                return DiagnosisCategorizationResult(
                    diagnoses=diagnoses, error=f"Category LLM error: {exc}"
                )
            if len(output.categories) == len(diagnoses):
                return DiagnosisCategorizationResult(
                    diagnoses=[
                        _apply(d, item)
                        for d, item in zip(diagnoses, output.categories, strict=True)
                    ],
                    reasoning=_join_reasons(output),
                )
            logger.warning(
                "Category count mismatch on attempt %d/%d: expected %d, got %d",
                attempt,
                attempts,
                len(diagnoses),
                len(output.categories),
            )
        matched = _match_by_name(diagnoses, output.categories)
        uncategorized = sum(item is None for item in matched)
        warning = count_mismatch_warning(
            len(diagnoses), len(output.categories), uncategorized, attempts
        )
        logger.warning(warning)
        return DiagnosisCategorizationResult(
            diagnoses=[_apply(d, item) for d, item in zip(diagnoses, matched, strict=True)],
            reasoning=_join_reasons(output),
            warnings=[warning],
        )

    async def _call_once(
        self, diagnoses: list[Diagnosis], patient_age: str | None
    ) -> DiagnosisCategorizationOutput:
        """One categorization call (with the chain's own transport/parse retries)."""
        output: DiagnosisCategorizationOutput = await self._chain.ainvoke(
            {"diagnoses": _format_diagnoses(diagnoses), "patient_age": patient_age or "unknown"}
        )
        logger.debug(
            "Categorized %d diagnoses reasoning=%r", len(output.categories), _join_reasons(output)
        )
        return output


class CategorizedDiagnosisExtractor:
    """Extract diagnoses from a note, then bucket them: the full ``DiagnosisExtractor``."""

    def __init__(
        self, extractor: DiagnosisExtractionClassifier, categorizer: DiagnosisCategorizer
    ) -> None:
        self._extractor = extractor
        self._categorizer = categorizer

    async def extract_diagnoses(
        self, metadata: DocumentMetadata, patient_age: str | None = None
    ) -> DiagnosisExtractionResult:
        """Run extraction then categorization; an extraction error skips categorization.

        ``patient_age`` (e.g. "3 months", from the patient-age step) only feeds categorization.
        """
        extracted = await self._extractor.extract_diagnoses(metadata)
        if extracted.error is not None:
            return extracted
        categorized = await self._categorizer.categorize(extracted.diagnoses, patient_age)
        reasoning = "\n".join(
            f"{label}: {text}"
            for label, text in (
                ("extraction", extracted.reasoning),
                ("categories", categorized.reasoning),
            )
            if text
        )
        return extracted.model_copy(
            update={
                "diagnoses": categorized.diagnoses,
                "diagnosis_categories": unique_categories(categorized.diagnoses),
                "reasoning": reasoning or None,
                "error": categorized.error,
                "warnings": [*extracted.warnings, *categorized.warnings],
            }
        )

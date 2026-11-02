"""Bucketing step: put each already-extracted diagnosis into a DiagnosisCategory.

Runs separately from extraction so the bucketing prompt can be tuned on plain diagnosis
strings, without a note. ``CategorizedDiagnosisExtractor`` chains the two steps.
"""

import logging

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

logger = logging.getLogger(__name__)


class CategorizedItem(BaseModel):
    """One diagnosis echoed back with its bucket."""

    name: str
    # Required and before category: makes the model reason per item (e.g. weigh the age)
    # even with thinking off, instead of picking the bucket in one step.
    reason: str
    category: DiagnosisCategory


class DiagnosisCategorizationOutput(BaseModel):
    """Structured LLM output for the categorization task."""

    # Declared first so the model reasons before picking buckets.
    reasoning: str | None = None
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

        ``patient_age`` (e.g. "3 months", from the patient-age step) decides child-vs-adult
        buckets; None is shown to the model as "unknown".
        """
        if not diagnoses:
            return DiagnosisCategorizationResult()
        try:
            output: DiagnosisCategorizationOutput = await self._chain.ainvoke(
                {
                    "diagnoses": _format_diagnoses(diagnoses),
                    "patient_age": patient_age or "unknown",
                }
            )
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
        reasoning = _join_reasons(output)
        logger.debug("Categorized %d diagnoses reasoning=%r", len(output.categories), reasoning)
        if len(output.categories) != len(diagnoses):
            error = (
                f"Category count mismatch: expected {len(diagnoses)}, got {len(output.categories)}"
            )
            logger.error(error)
            return DiagnosisCategorizationResult(
                diagnoses=diagnoses, reasoning=reasoning, error=error
            )
        return DiagnosisCategorizationResult(
            diagnoses=[
                d.model_copy(update={"category": item.category, "category_reason": item.reason})
                for d, item in zip(diagnoses, output.categories, strict=True)
            ],
            reasoning=reasoning,
        )


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
            }
        )

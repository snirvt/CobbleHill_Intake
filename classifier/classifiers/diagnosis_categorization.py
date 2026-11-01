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

from classifier.classifiers.categories import CATEGORY_LIST, CATEGORY_RULES
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

Return exactly one item per diagnosis, in the same order as the numbered list, echoing \
its name. Give a brief reasoning for any diagnosis whose category is not obvious.

{format_instructions}

Patient age: {patient_age}

--- DIAGNOSES ---
{diagnoses}
--- END ---"""


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
        logger.debug(
            "Categorized %d diagnoses reasoning=%r", len(output.categories), output.reasoning
        )
        if len(output.categories) != len(diagnoses):
            error = (
                f"Category count mismatch: expected {len(diagnoses)}, got {len(output.categories)}"
            )
            logger.error(error)
            return DiagnosisCategorizationResult(
                diagnoses=diagnoses, reasoning=output.reasoning, error=error
            )
        return DiagnosisCategorizationResult(
            diagnoses=[
                d.model_copy(update={"category": item.category})
                for d, item in zip(diagnoses, output.categories, strict=True)
            ],
            reasoning=output.reasoning,
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

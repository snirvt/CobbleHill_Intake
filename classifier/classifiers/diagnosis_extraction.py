import logging

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from classifier.classifiers.retry import with_llm_retry
from classifier.models import Diagnosis, DiagnosisExtractionResult, DocumentMetadata
from classifier.providers.structured import bind_output_schema

logger = logging.getLogger(__name__)


class DiagnosisExtractionOutput(BaseModel):
    """Structured LLM output for the diagnosis-extraction task."""

    # Declared first so the model reasons before listing diagnoses.
    reasoning: str | None = None
    # Required: JSON without this key is a parse error, not "no diagnoses".
    diagnoses: list[Diagnosis]


_PROMPT_TEMPLATE = """\
You are a medical records extraction assistant. Read the note below and extract every \
distinct diagnosis the patient has for this encounter.

Where to take the diagnoses from, in priority order:
1. The doctor's/medical professional's own conclusions: the Assessment, Impression, \
Diagnosis, or Assessment & Plan section, or diagnoses with ICD codes.
2. Only if the note has no clinician diagnosis at all, fall back to the condition(s) \
described in the HPI or chief complaint.

The clinician's diagnosis always wins over the patient's or family's complaint. \
Complaints and symptoms (e.g. "cough", "fever", "fussy", "not eating") are NOT \
diagnoses when the doctor names a condition — e.g. chief complaint "cough and fever" \
with assessment "community-acquired pneumonia" → "Community-acquired pneumonia".

Rules:
- Return each diagnosis name as written in the note (keep laterality, severity and \
other qualifiers), without surrounding punctuation.
- Include the ICD-10 code only if it is explicitly present in the note; otherwise null.
- Do not duplicate the same diagnosis, and do not invent diagnoses or ICD codes.
- Return an empty list only if the note has no clinician diagnosis and no condition \
described in the HPI or chief complaint.
- Give a brief reasoning citing the relevant wording and which section each \
diagnosis came from.

{format_instructions}

--- DOCTOR NOTE ---
{dr_full_text}
--- END ---"""


class DiagnosisExtractionClassifier:
    """Extracts diagnoses (name + optional ICD code) from a dr note's raw text."""

    def __init__(self, llm: BaseChatModel) -> None:
        parser: PydanticOutputParser[DiagnosisExtractionOutput] = PydanticOutputParser(
            pydantic_object=DiagnosisExtractionOutput
        )
        prompt = ChatPromptTemplate.from_template(_PROMPT_TEMPLATE).partial(
            format_instructions=parser.get_format_instructions()
        )
        self._chain = with_llm_retry(
            prompt | bind_output_schema(llm, DiagnosisExtractionOutput) | parser
        )

    async def extract_diagnoses(
        self, metadata: DocumentMetadata
    ) -> DiagnosisExtractionResult:
        """Return the diagnoses extracted from the dr note; empty list on failure."""
        try:
            output: DiagnosisExtractionOutput = await self._chain.ainvoke(
                {"dr_full_text": metadata.raw_text or "None"}
            )
            logger.debug(
                "Extracted %d diagnoses from %s reasoning=%r",
                len(output.diagnoses),
                metadata.file_path,
                output.reasoning,
            )
            return DiagnosisExtractionResult(
                file_path=metadata.file_path,
                diagnoses=output.diagnoses,
                reasoning=output.reasoning,
            )
        except OutputParserException as exc:
            logger.error("Parse failed for %s: %s", metadata.file_path, exc)
            return DiagnosisExtractionResult(
                file_path=metadata.file_path,
                diagnoses=[],
                reasoning=f"Parse error: {exc}",
                error=f"Parse error: {exc}",
            )
        except Exception as exc:
            logger.error("LLM call failed for %s: %s", metadata.file_path, exc)
            return DiagnosisExtractionResult(
                file_path=metadata.file_path,
                diagnoses=[],
                reasoning=f"LLM error: {exc}",
                error=f"LLM error: {exc}",
            )

import logging
from typing import Optional

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from classifier.models import DocumentMetadata, TreatmentRequestResult

logger = logging.getLogger(__name__)


class TreatmentRequestOutput(BaseModel):
    """Structured LLM output for the treatment-request task."""

    treatment_requested: bool
    reasoning: Optional[str] = None


_PROMPT_TEMPLATE = """\
You are a medical records auditor. Read the doctor's note below and decide whether \
ANYONE indicates the patient should receive treatment — this includes the patient \
requesting it, a family member/caregiver requesting it, or a clinician/provider \
recommending or ordering it (e.g. a procedure, medication, referral, or intervention).

Rules:
- Answer true if any such request or recommendation for treatment is present, from any source.
- Answer false only when the note contains no indication that the patient should be treated.
- Give a brief reasoning citing the relevant wording when present.

{format_instructions}

--- DOCTOR NOTE ---
{dr_full_text}
--- END ---"""


class TreatmentRequestClassifier:
    """Determines whether anyone indicates the patient should receive treatment, from a dr note."""

    def __init__(self, llm: BaseChatModel) -> None:
        parser: PydanticOutputParser[TreatmentRequestOutput] = PydanticOutputParser(
            pydantic_object=TreatmentRequestOutput
        )
        prompt = ChatPromptTemplate.from_template(_PROMPT_TEMPLATE).partial(
            format_instructions=parser.get_format_instructions()
        )
        self._chain = prompt | llm | parser

    async def extract_treatment_request(
        self, metadata: DocumentMetadata
    ) -> TreatmentRequestResult:
        """Return whether anyone indicates treatment is needed; defaults to False on failure."""
        try:
            output: TreatmentRequestOutput = await self._chain.ainvoke(
                {"dr_full_text": metadata.raw_text or "None"}
            )
            logger.debug(
                "Treatment requested=%s for %s",
                output.treatment_requested,
                metadata.file_path,
            )
            return TreatmentRequestResult(
                file_path=metadata.file_path,
                treatment_requested=output.treatment_requested,
                reasoning=output.reasoning,
            )
        except OutputParserException as exc:
            logger.error("Parse failed for %s: %s", metadata.file_path, exc)
            return TreatmentRequestResult(
                file_path=metadata.file_path,
                treatment_requested=False,
                reasoning=f"Parse error: {exc}",
            )
        except Exception as exc:
            logger.error("LLM call failed for %s: %s", metadata.file_path, exc)
            return TreatmentRequestResult(
                file_path=metadata.file_path,
                treatment_requested=False,
                reasoning=f"LLM error: {exc}",
            )

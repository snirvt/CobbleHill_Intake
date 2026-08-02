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
You are a medical records auditor. Read the note(s) below and decide whether the \
patient needs care or treatment. This is true in either of two ways:

1. REQUEST — anyone indicates the patient should receive treatment: the patient \
requesting it, a family member/caregiver requesting it, or a clinician/provider \
recommending or ordering it (e.g. a procedure, medication, referral, or intervention).
2. MEDICAL NECESSITY — the documented clinical situation itself requires care \
(e.g. an active diagnosis, abnormal findings, or ongoing condition being managed), \
even when nobody explicitly requests treatment.

Rules:
- Answer true if EITHER a request or a medical necessity is present, from any source or note.
- Answer false only when the note(s) contain no indication that the patient needs care.
- Give a brief reasoning citing the relevant wording, and say which of the two grounds applies.

{format_instructions}

--- NOTE(S) ---
{note_text}
--- END ---"""


class TreatmentRequestClassifier:
    """Determines whether a note's text shows the patient needs care.

    True on either ground: someone requested/recommended treatment, or the documented
    clinical situation itself requires care. Works on any note text — a single dr note
    or several notes concatenated.
    """

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
        """Return whether the patient needs care; defaults to False on failure."""
        try:
            output: TreatmentRequestOutput = await self._chain.ainvoke(
                {"note_text": metadata.raw_text or "None"}
            )
            logger.debug(
                "Care needed=%s for %s",
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

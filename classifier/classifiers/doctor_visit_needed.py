import logging
from typing import Literal

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from classifier.models import DocumentMetadata

logger = logging.getLogger(__name__)


class DoctorVisitOutput(BaseModel):
    """Structured LLM output for the doctor-visit-needed classification task."""

    verdict: Literal["NEEDS_VISIT", "NO_VISIT_NEEDED"]


_PROMPT_TEMPLATE = """\
You are a medical triage assistant. Based on the clinical notes below, decide whether \
this patient needs to be seen by a doctor.

{format_instructions}

--- CLINICAL NOTES ---
Chief Complaints: {complaints}
HPI: {hpi}
Review of Systems: {ros}
Assessment / Diagnoses: {assessment}
Plan: {plan}
Medications: {medications}
--- END ---"""


class DoctorVisitNeededClassifier:
    """Task classifier: determines whether a doctor visit is needed."""

    task_name = "doctor_visit_needed"

    def __init__(self, llm: BaseChatModel) -> None:
        parser: PydanticOutputParser[DoctorVisitOutput] = PydanticOutputParser(
            pydantic_object=DoctorVisitOutput
        )
        prompt = ChatPromptTemplate.from_template(_PROMPT_TEMPLATE).partial(
            format_instructions=parser.get_format_instructions()
        )
        self._chain = prompt | llm | parser

    def _build_prompt_input(self, metadata: DocumentMetadata) -> dict[str, str]:
        def fmt_list(items: list[str]) -> str:
            return "; ".join(items) if items else "None"

        def fmt_meds(m: DocumentMetadata) -> str:
            if not m.medications:
                return "None"
            parts: list[str] = []
            if m.medications.taking:
                parts.append("Taking: " + ", ".join(m.medications.taking))
            if m.medications.not_taking:
                parts.append("Not taking: " + ", ".join(m.medications.not_taking))
            return " | ".join(parts) or "None"

        return {
            "complaints": fmt_list(metadata.complaints),
            "hpi": metadata.hpi or "None",
            "ros": fmt_list(metadata.ros),
            "assessment": fmt_list(metadata.assessment),
            "plan": metadata.plan or "None",
            "medications": fmt_meds(metadata),
        }

    async def run(self, metadata: DocumentMetadata) -> str:
        """Return NEEDS_VISIT or NO_VISIT_NEEDED for the given document."""
        try:
            output: DoctorVisitOutput = await self._chain.ainvoke(
                self._build_prompt_input(metadata)
            )
            return output.verdict
        except OutputParserException as exc:
            logger.warning("Parse failed, defaulting to NEEDS_VISIT: %s", exc)
            return "NEEDS_VISIT"

import logging
import re
from typing import Literal, Optional

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from classifier.models import (
    IdentityMatchResult,
    PairClassificationResult,
    PairDocumentMetadata,
)

logger = logging.getLogger(__name__)


class ClinicalMatchOutput(BaseModel):
    """Structured LLM output for the clinical-match classification task."""

    verdict: Literal["MATCH", "MISMATCH", "UNDECIDED"]
    reasoning: Optional[str] = None


_PROMPT_TEMPLATE = """\
You are a medical records auditor. Compare the dr progress note and the nurse visit note \
below and determine whether they describe the same patient encounter consistently.

Output instructions:
{format_instructions}

--- DR PROGRESS NOTE ---
Chief Complaints: {dr_complaints}
Assessment / Diagnoses: {dr_assessment}
Plan: {dr_plan}
Medications: {dr_medications}

--- NURSE VISIT NOTE ---
Chief Complaint: {nurse_complaint}
Diagnoses: {nurse_diagnoses}
Assessment: {nurse_assessment}
Plan: {nurse_plan}
--- END ---"""


class DrNurseMatchClassifier:
    """Compares a dr progress note with a nurse visit note for patient and clinical consistency."""

    def __init__(self, llm: BaseChatModel) -> None:
        parser: PydanticOutputParser[ClinicalMatchOutput] = PydanticOutputParser(
            pydantic_object=ClinicalMatchOutput
        )
        prompt = ChatPromptTemplate.from_template(_PROMPT_TEMPLATE).partial(
            format_instructions=parser.get_format_instructions()
        )
        self._chain = prompt | llm | parser

    async def classify_pair(self, pair: PairDocumentMetadata) -> PairClassificationResult:
        identity = self._compare_identity(pair)
        clinical_verdict, clinical_reasoning = await self._compare_clinical(pair)

        overall = clinical_verdict

        return PairClassificationResult(
            dr_file_path=pair.dr_file_path,
            nurse_file_path=pair.nurse_file_path,
            identity_match=identity,
            clinical_verdict=clinical_verdict,
            clinical_reasoning=clinical_reasoning,
            overall=overall,
            dr_metadata=pair.dr,
            nurse_fields=pair.nurse,
        )

    def _compare_identity(self, pair: PairDocumentMetadata) -> IdentityMatchResult:
        dr = pair.dr.meta
        nurse = pair.nurse.meta
        return IdentityMatchResult(
            patient_name=self._names_match(dr.patient_name, nurse.patient_name),
            dob=self._dates_match(dr.dob, nurse.dob),
            dos=self._dates_match(dr.dos, nurse.dos),
            sex=self._sex_match(dr.sex, nurse.sex),
            account_number=self._tokens_match(dr.account_number, nurse.prn),
        )

    def _build_prompt_input(self, pair: PairDocumentMetadata) -> dict[str, str]:
        dr = pair.dr
        nurse = pair.nurse

        def fmt(items: list[str]) -> str:
            return "; ".join(items) if items else "None"

        def fmt_meds(meta: object) -> str:
            meds = getattr(meta, "medications", None)
            if not meds:
                return "None"
            parts: list[str] = []
            if meds.taking:
                parts.append("Taking: " + ", ".join(meds.taking))
            if meds.not_taking:
                parts.append("Not taking: " + ", ".join(meds.not_taking))
            return " | ".join(parts) or "None"

        return {
            "dr_complaints": fmt(dr.complaints),
            "dr_assessment": fmt(dr.assessment),
            "dr_plan": dr.plan or "None",
            "dr_medications": fmt_meds(dr),
            "nurse_complaint": nurse.chief_complaint or "None",
            "nurse_diagnoses": fmt(nurse.diagnoses),
            "nurse_assessment": fmt(nurse.assessment),
            "nurse_plan": nurse.plan or "None",
        }

    async def _compare_clinical(self, pair: PairDocumentMetadata) -> tuple[str, str]:
        try:
            output: ClinicalMatchOutput = await self._chain.ainvoke(
                self._build_prompt_input(pair)
            )
            logger.debug("Pair clinical verdict=%s", output.verdict)
            return output.verdict, output.reasoning
        except OutputParserException as exc:
            logger.error("Parse failed for pair %s/%s: %s", pair.dr_file_path, pair.nurse_file_path, exc)
            return "MISMATCH", f"Parse error: {exc}"
        except Exception as exc:
            logger.error("LLM call failed for pair %s/%s: %s", pair.dr_file_path, pair.nurse_file_path, exc)
            return "MISMATCH", f"LLM error: {exc}"

    @staticmethod
    def _normalize(s: str | None) -> str:
        if not s:
            return ""
        return re.sub(r"[\s,.']+", " ", s).strip().lower()

    @staticmethod
    def _normalize_date(s: str | None) -> str:
        """Normalize date to MM/DD/YYYY — handles 2-digit year."""
        if not s:
            return ""
        parts = re.split(r"[/\-]", s.strip())
        if len(parts) == 3:
            m, d, y = parts
            if len(y) == 2:
                y = ("20" + y) if int(y) < 50 else ("19" + y)
            return f"{m.zfill(2)}/{d.zfill(2)}/{y}"
        return s.strip()

    def _names_match(self, a: str | None, b: str | None) -> bool:
        return bool(a and b and self._normalize(a) == self._normalize(b))

    def _dates_match(self, a: str | None, b: str | None) -> bool:
        return bool(a and b and self._normalize_date(a) == self._normalize_date(b))

    def _sex_match(self, a: str | None, b: str | None) -> bool:
        if not a or not b:
            return False
        _to_letter = {"male": "m", "female": "f", "m": "m", "f": "f"}
        return _to_letter.get(a.strip().lower()) == _to_letter.get(b.strip().lower())

    def _tokens_match(self, a: str | None, b: str | None) -> bool:
        return bool(a and b and a.strip().upper() == b.strip().upper())

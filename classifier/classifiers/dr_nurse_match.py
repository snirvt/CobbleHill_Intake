import asyncio
import logging
import re
from typing import Optional

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from classifier.models import (
    CareCheck,
    ClinicalVerdict,
    Diagnosis,
    DiagnosisCheck,
    DocumentMetadata,
    IdentityMatchResult,
    PairClassificationResult,
    PairDocumentMetadata,
    PatientMetadata,
)
from classifier.protocols import DiagnosisExtractor, TreatmentRequestExtractor
from config.settings import settings

logger = logging.getLogger(__name__)


class ClinicalMatchOutput(BaseModel):
    """Structured LLM output for the clinical-match classification task."""

    verdict: ClinicalVerdict
    reasoning: Optional[str] = None


_PROMPT_TEMPLATE = """\
You are a medical records auditor. Compare the dr note and the nurse visit note \
below and determine whether they describe the same patient and conditions consistently.

Keep in mind the dates of the meeting may be different (hence the patient age could be slightly different in each note), it's up to you to decide if there is a match.

Output instructions:
{format_instructions}

--- DR PROGRESS NOTE ---
{dr_full_text}

--- NURSE VISIT NOTE ---
{nurse_full_text}
--- END ---"""


class DrNurseMatchClassifier:
    """Compares a dr progress note with a nurse visit note for patient and clinical consistency."""

    def __init__(
        self,
        llm: BaseChatModel,
        diagnosis_extractor: DiagnosisExtractor | None = None,
        care_extractor: TreatmentRequestExtractor | None = None,
    ) -> None:
        parser: PydanticOutputParser[ClinicalMatchOutput] = PydanticOutputParser(
            pydantic_object=ClinicalMatchOutput
        )
        prompt = ChatPromptTemplate.from_template(_PROMPT_TEMPLATE).partial(
            format_instructions=parser.get_format_instructions()
        )
        self._chain = prompt | llm | parser
        self._diagnosis_extractor = diagnosis_extractor
        self._care_extractor = care_extractor

    async def classify_pair(self, pair: PairDocumentMetadata) -> PairClassificationResult:
        identity = self._compare_identity(pair)
        (
            (clinical_verdict, clinical_reasoning),
            (diagnosis_check, diagnoses),
            (care_check, care_reasoning),
        ) = await asyncio.gather(
            self._compare_clinical(pair),
            self._check_diagnoses(pair),
            self._check_care_needed(pair),
        )

        overall = clinical_verdict
        if diagnosis_check is DiagnosisCheck.MISSING or care_check is CareCheck.NOT_NEEDED:
            overall = ClinicalVerdict.MISMATCH

        return PairClassificationResult(
            dr_file_path=pair.dr_file_path,
            nurse_file_path=pair.nurse_file_path,
            identity_match=identity,
            clinical_verdict=clinical_verdict,
            clinical_reasoning=clinical_reasoning,
            overall=overall,
            dr_metadata=pair.dr,
            nurse_fields=pair.nurse,
            diagnosis_check=diagnosis_check,
            diagnoses=diagnoses,
            care_check=care_check,
            care_reasoning=care_reasoning,
        )

    async def _check_care_needed(
        self, pair: PairDocumentMetadata
    ) -> tuple[CareCheck | None, str]:
        """Decide whether the pair shows the patient needs care, from both notes.

        Care counts as needed on either ground: someone requested/recommended it, or
        the clinical situation itself requires it. Returns (None, "") when the pair's
        category is out of scope or no care extractor was injected.
        """
        if self._care_extractor is None:
            return None, ""
        if pair.category not in settings.care_check_categories:
            return None, ""

        combined = DocumentMetadata(
            file_path=pair.dr_file_path,
            raw_text=self._combined_note_text(pair),
            meta=pair.dr.meta,
        )
        result = await self._care_extractor.extract_treatment_request(combined)
        check = CareCheck.NEEDED if result.treatment_requested else CareCheck.NOT_NEEDED
        if check is CareCheck.NOT_NEEDED:
            logger.info(
                "No care need found for %s pair %s / %s",
                pair.category,
                pair.dr_file_path,
                pair.nurse_file_path,
            )
        return check, result.reasoning or ""

    @staticmethod
    def _combined_note_text(pair: PairDocumentMetadata) -> str:
        """Both notes in one text block, each under a role header."""
        return (
            f"--- DR NOTE(S) ---\n{pair.dr.raw_text or 'None'}\n\n"
            f"--- NURSE NOTE(S) ---\n{pair.nurse_raw_text or 'None'}"
        )

    async def _check_diagnoses(
        self, pair: PairDocumentMetadata
    ) -> tuple[DiagnosisCheck | None, list[Diagnosis]]:
        """Extract diagnoses from both notes; MISSING when neither note has any.

        Returns (None, []) when the pair's category is out of scope for the check
        or no diagnosis extractor was injected — the check is then skipped entirely.
        """
        if self._diagnosis_extractor is None:
            return None, []
        if pair.category not in settings.diagnosis_check_categories:
            return None, []

        nurse_doc = DocumentMetadata(
            file_path=pair.nurse_file_path,
            raw_text=pair.nurse_raw_text,
            meta=PatientMetadata(),
        )
        dr_result, nurse_result = await asyncio.gather(
            self._diagnosis_extractor.extract_diagnoses(pair.dr),
            self._diagnosis_extractor.extract_diagnoses(nurse_doc),
        )
        diagnoses = [
            d.model_copy(update={"source": source})
            for source, result in (("dr", dr_result), ("nurse", nurse_result))
            for d in result.diagnoses
        ]
        check = DiagnosisCheck.EXISTS if diagnoses else DiagnosisCheck.MISSING
        if check is DiagnosisCheck.MISSING:
            logger.info(
                "No diagnosis found for %s pair %s / %s",
                pair.category,
                pair.dr_file_path,
                pair.nurse_file_path,
            )
        return check, diagnoses

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
        return {
            "dr_full_text": pair.dr.raw_text or "None",
            "nurse_full_text": pair.nurse_raw_text or "None",
        }

    async def _compare_clinical(self, pair: PairDocumentMetadata) -> tuple[ClinicalVerdict, str]:
        try:
            output: ClinicalMatchOutput = await self._chain.ainvoke(
                self._build_prompt_input(pair)
            )
            logger.debug("Pair clinical verdict=%s", output.verdict)
            return output.verdict, output.reasoning or ""
        except OutputParserException as exc:
            logger.error("Parse failed for pair %s/%s: %s", pair.dr_file_path, pair.nurse_file_path, exc)
            return ClinicalVerdict.MISMATCH, f"Parse error: {exc}"
        except Exception as exc:
            logger.error("LLM call failed for pair %s/%s: %s", pair.dr_file_path, pair.nurse_file_path, exc)
            return ClinicalVerdict.MISMATCH, f"LLM error: {exc}"

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

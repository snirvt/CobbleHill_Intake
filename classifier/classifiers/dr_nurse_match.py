import logging
import re

from classifier.models import (
    IdentityMatchResult,
    PairClassificationResult,
    PairDocumentMetadata,
)
from classifier.protocols import LLMProvider

logger = logging.getLogger(__name__)

_VERDICTS = frozenset({"MATCH", "PARTIAL_MATCH", "MISMATCH"})

_PROMPT_TEMPLATE = """\
You are a medical records auditor. Compare the dr progress note and the nurse visit note \
below and determine whether they describe the same patient encounter consistently.

Respond with EXACTLY one verdict on the first line, then a brief explanation (2-4 sentences):
  MATCH          — all key clinical details align
  PARTIAL_MATCH  — some details match, minor discrepancies exist
  MISMATCH       — significant clinical inconsistencies or wrong patient

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
--- END ---

Verdict and explanation:"""


class DrNurseMatchClassifier:
    """Compares a dr progress note with a nurse visit note for patient and clinical consistency."""

    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    async def classify_pair(self, pair: PairDocumentMetadata) -> PairClassificationResult:
        identity = self._compare_identity(pair)
        clinical_verdict, clinical_reasoning = await self._compare_clinical(pair)

        # Overall: if identity fully fails, demote to MISMATCH regardless of LLM
        identity_score = sum([
            identity.patient_name,
            identity.dob,
            identity.dos,
            identity.sex,
        ])
        overall = clinical_verdict
        if identity_score == 0:
            overall = "MISMATCH"
        elif identity_score < 3 and clinical_verdict == "MATCH":
            overall = "PARTIAL_MATCH"

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

    async def _compare_clinical(self, pair: PairDocumentMetadata) -> tuple[str, str]:
        prompt = self._build_prompt(pair)
        try:
            response = await self._llm.complete(prompt)
        except Exception as exc:
            logger.error("LLM call failed for pair %s/%s: %s", pair.dr_file_path, pair.nurse_file_path, exc)
            return "MISMATCH", f"LLM error: {exc}"
        return self._parse_response(response)

    def _build_prompt(self, pair: PairDocumentMetadata) -> str:
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

        return _PROMPT_TEMPLATE.format(
            dr_complaints=fmt(dr.complaints),
            dr_assessment=fmt(dr.assessment),
            dr_plan=dr.plan or "None",
            dr_medications=fmt_meds(dr),
            nurse_complaint=nurse.chief_complaint or "None",
            nurse_diagnoses=fmt(nurse.diagnoses),
            nurse_assessment=fmt(nurse.assessment),
            nurse_plan=nurse.plan or "None",
        )

    def _parse_response(self, response: str) -> tuple[str, str]:
        lines = response.strip().splitlines()
        verdict = "MISMATCH"
        # Check longest tokens first so "MATCH" doesn't shadow "PARTIAL_MATCH"
        _ordered = ("PARTIAL_MATCH", "MISMATCH", "MATCH")
        for line in lines:
            upper = line.strip().upper()
            for v in _ordered:
                if v in upper:
                    verdict = v
                    break
            else:
                continue
            break
        reasoning = " ".join(line.strip() for line in lines[1:] if line.strip()) or response.strip()
        logger.debug("Pair clinical verdict=%s", verdict)
        return verdict, reasoning

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

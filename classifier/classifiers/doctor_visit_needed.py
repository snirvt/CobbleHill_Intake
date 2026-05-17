import logging

from classifier.models import DocumentMetadata
from classifier.protocols import LLMProvider

logger = logging.getLogger(__name__)

_CATEGORIES = frozenset({"NEEDS_VISIT", "NO_VISIT_NEEDED"})

_PROMPT_TEMPLATE = """\
You are a medical triage assistant. Based on the clinical notes below, decide whether \
this patient needs to be seen by a doctor.

Respond with EXACTLY one of these two words and nothing else:
  NEEDS_VISIT       — patient has an active complaint, abnormal finding, or unresolved issue
  NO_VISIT_NEEDED   — routine well-visit with no concerns, or follow-up not required

--- CLINICAL NOTES ---
Chief Complaints: {complaints}
HPI: {hpi}
Review of Systems: {ros}
Assessment / Diagnoses: {assessment}
Plan: {plan}
Medications: {medications}
--- END ---

Answer (one word only):"""


class DoctorVisitNeededClassifier:
    """Task classifier: determines whether a doctor visit is needed."""

    task_name = "doctor_visit_needed"

    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    def _build_prompt(self, metadata: DocumentMetadata) -> str:
        def fmt_list(items: list[str]) -> str:
            return "; ".join(items) if items else "None"

        def fmt_meds(metadata: DocumentMetadata) -> str:
            if not metadata.medications:
                return "None"
            parts: list[str] = []
            if metadata.medications.taking:
                parts.append("Taking: " + ", ".join(metadata.medications.taking))
            if metadata.medications.not_taking:
                parts.append("Not taking: " + ", ".join(metadata.medications.not_taking))
            return " | ".join(parts) or "None"

        return _PROMPT_TEMPLATE.format(
            complaints=fmt_list(metadata.complaints),
            hpi=metadata.hpi or "None",
            ros=fmt_list(metadata.ros),
            assessment=fmt_list(metadata.assessment),
            plan=metadata.plan or "None",
            medications=fmt_meds(metadata),
        )

    def _parse_category(self, response: str) -> str:
        """Extract category from LLM response; default to NEEDS_VISIT if ambiguous."""
        upper = response.strip().upper()
        for cat in _CATEGORIES:
            if cat in upper:
                return cat
        logger.warning("Unexpected LLM response %r — defaulting to NEEDS_VISIT", response)
        return "NEEDS_VISIT"

    async def run(self, metadata: DocumentMetadata) -> str:
        """Return NEEDS_VISIT or NO_VISIT_NEEDED for the given document."""
        prompt = self._build_prompt(metadata)
        response = await self._llm.complete(prompt)
        return self._parse_category(response)

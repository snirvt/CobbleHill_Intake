import logging
import re

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from classifier.models import DocumentMetadata, MainDiagnosisResult
from config.settings import settings

logger = logging.getLogger(__name__)


class MainDiagnosisOutput(BaseModel):
    """Structured LLM output for the main-diagnosis task."""

    main_diagnosis: str | None = None
    reasoning: str | None = None


_PROMPT_TEMPLATE = """\
You are a medical records assistant. Read the doctor's note below and identify the \
patient's MAIN diagnosis — the condition the doctor/medical professional concluded \
the patient has. The label is used to group similar cases together, so it must be \
short and generic.

Where to take the diagnosis from, in priority order:
1. The doctor's/medical professional's own conclusion: the Assessment, Impression, \
Diagnosis, or Assessment & Plan section, or diagnoses with ICD codes.
2. Only if the note has no clinician diagnosis at all, fall back to the condition \
described in the HPI or chief complaint.

The clinician's diagnosis always wins over the patient's or family's complaint. \
Complaints and symptoms (e.g. "cough", "fever", "fussy", "not eating") are NOT the \
main diagnosis when the doctor names a condition — e.g. chief complaint "cough and \
fever" with assessment "community-acquired pneumonia" → "Pneumonia".

Rules:
- Pick exactly one diagnosis: the doctor's primary diagnosis for this encounter (if \
several are listed, the first/primary one).
- Answer in 1-2 words, using the general condition name (e.g. "Pneumonia", \
"Otitis Media", "Asthma", "UTI", "Fracture").
- Prefer the shortest common name or a standard abbreviation over a long name.
- Do not include ICD codes, laterality, severity, stage, or other qualifiers.
- If the note contains no diagnosis, return null for main_diagnosis.
- Give a brief reasoning citing the relevant wording and which section it came from.

{format_instructions}

--- DOCTOR NOTE ---
{dr_full_text}
--- END ---"""

# ICD-10 codes with a decimal part (e.g. J18.9), or a bare code in parentheses (e.g. "(J18)").
_ICD_PATTERN = re.compile(r"\(?\b[A-TV-Z]\d{2}\.[0-9A-Z]{1,4}\b\)?|\([A-TV-Z]\d{2}\)")
# Anything other than letters, digits, whitespace, hyphens and apostrophes.
_PUNCT_PATTERN = re.compile(r"[^\w\s'-]|_")
# LLM answers that mean "no diagnosis".
_EMPTY_LABELS = {"none", "n/a", "na", "null", "unknown", "no diagnosis"}
# All-caps words up to this length are kept as acronyms (UTI, COPD); longer ones are title-cased.
_ACRONYM_MAX_LEN = 4


class MainDiagnosisClassifier:
    """Extracts a short (1-2 word) main-diagnosis label from a dr note, for grouping cases."""

    def __init__(self, llm: BaseChatModel) -> None:
        parser: PydanticOutputParser[MainDiagnosisOutput] = PydanticOutputParser(
            pydantic_object=MainDiagnosisOutput
        )
        prompt = ChatPromptTemplate.from_template(_PROMPT_TEMPLATE).partial(
            format_instructions=parser.get_format_instructions()
        )
        self._chain = prompt | llm | parser

    @staticmethod
    def _normalize(label: str | None) -> str | None:
        """Turn a raw LLM label into a stable grouping key; None when it means no diagnosis.

        Drops ICD codes and punctuation, collapses whitespace, title-cases words (short
        all-caps acronyms kept), and truncates to ``settings.main_diagnosis_max_words``.
        """
        if label is None or label.strip(" .\"'").lower() in _EMPTY_LABELS:
            return None
        text = _ICD_PATTERN.sub(" ", label)
        text = _PUNCT_PATTERN.sub(" ", text)
        words = text.split()
        if not words or " ".join(words).lower() in _EMPTY_LABELS:
            return None
        words = [
            w if w.isupper() and len(w) <= _ACRONYM_MAX_LEN else w.capitalize()
            for w in words[: settings.main_diagnosis_max_words]
        ]
        return " ".join(words)

    async def extract_main_diagnosis(self, metadata: DocumentMetadata) -> MainDiagnosisResult:
        """Return the normalized main-diagnosis label; None on failure or no diagnosis."""
        try:
            output: MainDiagnosisOutput = await self._chain.ainvoke(
                {"dr_full_text": metadata.raw_text or "None"}
            )
            label = self._normalize(output.main_diagnosis)
            logger.debug(
                "Main diagnosis raw=%r normalized=%r for %s",
                output.main_diagnosis,
                label,
                metadata.file_path,
            )
            return MainDiagnosisResult(
                file_path=metadata.file_path,
                main_diagnosis=label,
                reasoning=output.reasoning,
            )
        except OutputParserException as exc:
            logger.error("Parse failed for %s: %s", metadata.file_path, exc)
            return MainDiagnosisResult(
                file_path=metadata.file_path, reasoning=f"Parse error: {exc}"
            )
        except Exception as exc:
            logger.error("LLM call failed for %s: %s", metadata.file_path, exc)
            return MainDiagnosisResult(
                file_path=metadata.file_path, reasoning=f"LLM error: {exc}"
            )

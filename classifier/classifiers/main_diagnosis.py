import logging
import re

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from classifier.classifiers.categories import CATEGORY_LIST, CATEGORY_RULES
from classifier.classifiers.retry import with_llm_retry
from classifier.models import DiagnosisCategory, DocumentMetadata, MainDiagnosisResult
from classifier.providers.structured import bind_output_schema
from config.settings import settings

logger = logging.getLogger(__name__)


class MainDiagnosisOutput(BaseModel):
    """Structured LLM output for the main-diagnosis task."""

    main_diagnosis: str | None = None
    diagnosis_category: DiagnosisCategory | None = None
    icd_code: str | None = None
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
- Answer with a short (usually 1-4 words) but complete condition name, using the \
general condition name (e.g. "Pneumonia", "Otitis Media", "Asthma", "UTI", "Fracture").
- Shorten by picking the core condition, never by cutting a phrase off after its first \
words. The label must make sense on its own and must never end on a connector word \
such as "for", "to", "of", "with", "due", "and" or "s/p".
- For administrative or encounter wording ("Encounter for ...", "Admission for ...", \
"Follow-up of ...", "Supervision of ...", "History of ..."), name the condition being \
cared for, not the encounter phrase — e.g. "Encounter for routine follow-up of \
hypertension" → "Hypertension".
- For "X due to / secondary to / s/p Y", name the main condition X, not the cause \
chain — e.g. "Acute kidney injury due to dehydration" → "AKI".
- Prefer the shortest common name or a standard abbreviation over a long name.
- Do not include ICD codes, laterality, severity, stage, or other qualifiers.
- If the note contains no diagnosis, return null for main_diagnosis.

Also put the main diagnosis into exactly one diagnosis_category from this list \
(copy the text exactly):
{categories}

Category rules:
{category_rules}
- If main_diagnosis is null, diagnosis_category is null too.

ICD code:
- icd_code: copy the ICD-10 code the note writes for this main diagnosis, exactly as \
written (e.g. "K21.9"). If the note gives no code for it, return null. Never guess or \
look up a code yourself.
- Give a brief reasoning citing the relevant wording and which section it came from.

{format_instructions}

--- DOCTOR NOTE ---
{dr_full_text}
--- END ---"""

# ICD-10 codes with a decimal part (e.g. J18.9), or a bare code in parentheses (e.g. "(J18)").
_ICD_PATTERN = re.compile(r"\(?\b[A-TV-Z]\d{2}\.[0-9A-Z]{1,4}\b\)?|\([A-TV-Z]\d{2}\)")
# A whole valid ICD-10 code, e.g. K21.9, R10.83, J18. Anything else is dropped.
_ICD_CODE_PATTERN = re.compile(r"[A-TV-Z]\d[0-9A-Z](\.[0-9A-Z]{1,4})?")
# Anything other than letters, digits, whitespace, hyphens and apostrophes.
_PUNCT_PATTERN = re.compile(r"[^\w\s'-]|_")
# LLM answers that mean "no diagnosis".
_EMPTY_LABELS = {"none", "n/a", "na", "null", "unknown", "no diagnosis"}
# All-caps words up to this length are kept as acronyms (UTI, COPD); longer ones are title-cased.
_ACRONYM_MAX_LEN = 4


class MainDiagnosisClassifier:
    """Extracts a short main-diagnosis label and its fixed category from a dr note."""

    def __init__(self, llm: BaseChatModel) -> None:
        parser: PydanticOutputParser[MainDiagnosisOutput] = PydanticOutputParser(
            pydantic_object=MainDiagnosisOutput
        )
        prompt = ChatPromptTemplate.from_template(_PROMPT_TEMPLATE).partial(
            format_instructions=parser.get_format_instructions(),
            categories=CATEGORY_LIST,
            category_rules=CATEGORY_RULES,
        )
        self._chain = with_llm_retry(prompt | bind_output_schema(llm, MainDiagnosisOutput) | parser)

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

    @staticmethod
    def _normalize_icd(code: str | None) -> str | None:
        """Uppercase and de-space an ICD-10 code; None when it is not a valid ICD-10 code."""
        if code is None:
            return None
        cleaned = "".join(code.split()).upper()
        if not _ICD_CODE_PATTERN.fullmatch(cleaned):
            if cleaned:
                logger.debug("Dropping malformed ICD code %r", code)
            return None
        return cleaned

    async def extract_main_diagnosis(self, metadata: DocumentMetadata) -> MainDiagnosisResult:
        """Return the normalized label, category and ICD code; None on failure or no diagnosis."""
        try:
            output: MainDiagnosisOutput = await self._chain.ainvoke(
                {"dr_full_text": metadata.raw_text or "None"}
            )
            label = self._normalize(output.main_diagnosis)
            icd_code = self._normalize_icd(output.icd_code) if label else None
            logger.debug(
                "Main diagnosis raw=%r normalized=%r category=%r icd=%r for %s",
                output.main_diagnosis,
                label,
                output.diagnosis_category,
                icd_code,
                metadata.file_path,
            )
            return MainDiagnosisResult(
                file_path=metadata.file_path,
                main_diagnosis=label,
                # A label that means "no diagnosis" carries no category either.
                diagnosis_category=output.diagnosis_category if label else None,
                icd_code=icd_code,
                reasoning=output.reasoning,
            )
        except OutputParserException as exc:
            logger.error("Parse failed for %s: %s", metadata.file_path, exc)
            return MainDiagnosisResult(
                file_path=metadata.file_path,
                reasoning=f"Parse error: {exc}",
                error=f"Parse error: {exc}",
            )
        except Exception as exc:
            logger.error("LLM call failed for %s: %s", metadata.file_path, exc)
            return MainDiagnosisResult(
                file_path=metadata.file_path,
                reasoning=f"LLM error: {exc}",
                error=f"LLM error: {exc}",
            )

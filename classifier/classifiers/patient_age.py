import logging
from datetime import date, datetime

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from classifier.models import DocumentMetadata, PatientAgeResult

logger = logging.getLogger(__name__)


class PatientAgeOutput(BaseModel):
    """Structured LLM output for the patient-age task."""

    age_years: float | None = None
    dob: str | None = None
    date_of_service: str | None = None
    reasoning: str | None = None


_PROMPT_TEMPLATE = """\
You are a medical records assistant. Read the doctor's note(s) below and find the \
patient's age.

Rules:
- age_years: the patient's age in years at the date of service, as a number. Convert \
months or weeks to a fraction of a year (e.g. "8 mo" = 0.67, "2 wk" = 0.04). Use null \
if no age is stated.
- dob: the patient's date of birth exactly as written, or null if absent.
- date_of_service: the visit/encounter date exactly as written, or null if absent.
- Do not guess. Only report values that appear in the note.
- Give a brief reasoning citing the relevant wording.

{format_instructions}

--- DOCTOR NOTE(S) ---
{dr_full_text}
--- END ---"""

# Date formats tried, in order, when computing age from DOB.
_DATE_FORMATS = ("%m/%d/%Y", "%m/%d/%y", "%Y-%m-%d", "%m-%d-%Y")
_DAYS_PER_YEAR = 365.25


def _parse_date(value: str | None) -> date | None:
    """Parse a date written in a common US/ISO format; None if unparseable."""
    if not value:
        return None
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(value.strip(), fmt).date()
        except ValueError:
            continue
    return None


class PatientAgeClassifier:
    """Reads the patient's age in years from dr note text (LLM + DOB fallback)."""

    def __init__(self, llm: BaseChatModel) -> None:
        parser: PydanticOutputParser[PatientAgeOutput] = PydanticOutputParser(
            pydantic_object=PatientAgeOutput
        )
        prompt = ChatPromptTemplate.from_template(_PROMPT_TEMPLATE).partial(
            format_instructions=parser.get_format_instructions()
        )
        self._chain = prompt | llm | parser

    @staticmethod
    def _compute_age(output: PatientAgeOutput, today: date | None = None) -> float | None:
        """Return the stated age; otherwise DOB to date of service (or today); else None."""
        if output.age_years is not None and output.age_years >= 0:
            return round(output.age_years, 2)
        dob = _parse_date(output.dob)
        if dob is None:
            return None
        as_of = _parse_date(output.date_of_service) or today or date.today()
        if as_of < dob:
            return None
        return round((as_of - dob).days / _DAYS_PER_YEAR, 2)

    async def extract_age(self, metadata: DocumentMetadata) -> PatientAgeResult:
        """Return the patient's age; age_years is None on failure or when not stated."""
        try:
            output: PatientAgeOutput = await self._chain.ainvoke(
                {"dr_full_text": metadata.raw_text or "None"}
            )
            age = self._compute_age(output)
            logger.debug("Patient age=%s for %s", age, metadata.file_path)
            return PatientAgeResult(
                file_path=metadata.file_path,
                age_years=age,
                dob=output.dob,
                date_of_service=output.date_of_service,
                reasoning=output.reasoning,
            )
        except OutputParserException as exc:
            logger.error("Parse failed for %s: %s", metadata.file_path, exc)
            return PatientAgeResult(file_path=metadata.file_path, reasoning=f"Parse error: {exc}")
        except Exception as exc:
            logger.error("LLM call failed for %s: %s", metadata.file_path, exc)
            return PatientAgeResult(file_path=metadata.file_path, reasoning=f"LLM error: {exc}")

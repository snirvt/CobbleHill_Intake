import logging

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from classifier.models import Diagnosis, DiagnosisExtractionResult, DocumentMetadata

logger = logging.getLogger(__name__)


class DiagnosisExtractionOutput(BaseModel):
    """Structured LLM output for the diagnosis-extraction task."""

    diagnoses: list[Diagnosis] = []


_PROMPT_TEMPLATE = """\
You are a medical records extraction assistant. Read the doctor's note below and \
extract every distinct diagnosis mentioned.

Rules:
- Return the diagnosis name as written (normalized, without surrounding punctuation).
- Include the ICD-10 code only if it is explicitly present in the note; otherwise leave it null.
- Do not invent diagnoses or ICD codes. If none are present, return an empty list.
- Do not duplicate the same diagnosis.

{format_instructions}

--- DOCTOR NOTE ---
{dr_full_text}
--- END ---"""


class DiagnosisExtractionClassifier:
    """Extracts diagnoses (name + optional ICD code) from a dr note's raw text."""

    def __init__(self, llm: BaseChatModel) -> None:
        parser: PydanticOutputParser[DiagnosisExtractionOutput] = PydanticOutputParser(
            pydantic_object=DiagnosisExtractionOutput
        )
        prompt = ChatPromptTemplate.from_template(_PROMPT_TEMPLATE).partial(
            format_instructions=parser.get_format_instructions()
        )
        self._chain = prompt | llm | parser

    async def extract_diagnoses(
        self, metadata: DocumentMetadata
    ) -> DiagnosisExtractionResult:
        """Return the diagnoses extracted from the dr note; empty list on failure."""
        try:
            output: DiagnosisExtractionOutput = await self._chain.ainvoke(
                {"dr_full_text": metadata.raw_text or "None"}
            )
            logger.debug(
                "Extracted %d diagnoses from %s",
                len(output.diagnoses),
                metadata.file_path,
            )
            return DiagnosisExtractionResult(
                file_path=metadata.file_path,
                diagnoses=output.diagnoses,
            )
        except OutputParserException as exc:
            logger.error("Parse failed for %s: %s", metadata.file_path, exc)
            return DiagnosisExtractionResult(file_path=metadata.file_path, diagnoses=[])
        except Exception as exc:
            logger.error("LLM call failed for %s: %s", metadata.file_path, exc)
            return DiagnosisExtractionResult(file_path=metadata.file_path, diagnoses=[])

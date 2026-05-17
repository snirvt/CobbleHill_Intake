from pathlib import Path
from pydantic import BaseModel


class ExtractedText(BaseModel):
    """Raw text extracted from a document file."""

    file_path: Path
    text: str
    num_pages: int


class PatientMetadata(BaseModel):
    """Patient-level fields extracted via regex from document text."""

    patient_name: str | None = None
    dob: str | None = None
    age: str | None = None
    sex: str | None = None
    account_number: str | None = None
    dos: str | None = None
    phone: str | None = None
    address: str | None = None


class ExaminationData(BaseModel):
    """Structured fields from the Examination section."""

    pediatric: list[str] = []
    general: str | None = None


class MedicationData(BaseModel):
    """Medications split by current taking status."""

    taking: list[str] = []
    not_taking: list[str] = []


class ExtractedFields(BaseModel):
    """All structured fields parsed from document text by a MetadataExtractor."""

    meta: PatientMetadata
    hpi: str | None = None
    examination: ExaminationData | None = None
    complaints: list[str] = []
    medical_history: list[str] = []
    surgical_history: list[str] = []
    hospitalization: list[str] = []
    assessment: list[str] = []
    ros: list[str] = []
    medications: MedicationData | None = None
    plan: str | None = None
    procedure_codes: list[str] = []
    preventive_medicine: str | None = None


class DocumentMetadata(BaseModel):
    """Full document context passed to the classifier."""

    file_path: Path
    raw_text: str
    meta: PatientMetadata
    hpi: str | None = None
    examination: ExaminationData | None = None
    complaints: list[str] = []
    medical_history: list[str] = []
    surgical_history: list[str] = []
    hospitalization: list[str] = []
    assessment: list[str] = []
    ros: list[str] = []
    medications: MedicationData | None = None
    plan: str | None = None
    procedure_codes: list[str] = []
    preventive_medicine: str | None = None


class ClassificationResult(BaseModel):
    """Output of a classifier for one document."""

    file_path: Path
    category: str
    task_results: dict[str, str] = {}
    metadata: DocumentMetadata


class PipelineResult(BaseModel):
    """Outcome of the full pipeline for one file."""

    file_path: Path
    success: bool
    result: ClassificationResult | None = None
    error: str | None = None

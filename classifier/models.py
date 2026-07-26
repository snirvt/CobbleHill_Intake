from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel


class ClinicalVerdict(StrEnum):
    MATCH = "MATCH"
    MISMATCH = "MISMATCH"
    UNDECIDED = "UNDECIDED"


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
    """All structured fields parsed from document text by a NoteExtractor."""

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


class Diagnosis(BaseModel):
    """A single diagnosis extracted from a dr note."""

    name: str
    icd_code: str | None = None


class DiagnosisExtractionResult(BaseModel):
    """All diagnoses extracted from one dr note."""

    file_path: Path
    diagnoses: list[Diagnosis] = []


class PipelineResult(BaseModel):
    """Outcome of the full pipeline for one file."""

    file_path: Path
    success: bool
    result: ClassificationResult | None = None
    error: str | None = None


class NursePatientMeta(BaseModel):
    """Patient identity/demographic fields from a nurse visit note."""

    patient_name: str | None = None
    dob: str | None = None
    age: str | None = None
    sex: str | None = None
    prn: str | None = None
    dos: str | None = None
    phone: str | None = None
    address: str | None = None
    seen_by: str | None = None


class NurseVisitFields(BaseModel):
    """Structured clinical fields parsed from a nurse visit SOAP note."""

    meta: NursePatientMeta = NursePatientMeta()
    chief_complaint: str | None = None
    diagnoses: list[str] = []
    medications_active: list[str] = []
    subjective: str | None = None
    objective: str | None = None
    assessment: list[str] = []
    plan: str | None = None
    care_plan: str | None = None


class IdentityMatchResult(BaseModel):
    """Per-field identity comparison between dr and nurse notes."""

    patient_name: bool = False
    dob: bool = False
    dos: bool = False
    sex: bool = False
    account_number: bool = False


class PairDocumentMetadata(BaseModel):
    """Both dr and nurse document metadata for pair classification."""

    dr_file_path: Path
    nurse_file_path: Path
    dr: DocumentMetadata
    nurse: NurseVisitFields
    nurse_raw_text: str = ""


class PairClassificationResult(BaseModel):
    """Output of pair classifier comparing dr and nurse notes."""

    dr_file_path: Path
    nurse_file_path: Path
    identity_match: IdentityMatchResult
    clinical_verdict: ClinicalVerdict
    clinical_reasoning: str
    overall: ClinicalVerdict
    dr_metadata: "DocumentMetadata"
    nurse_fields: NurseVisitFields


class PairPipelineResult(BaseModel):
    """Outcome of the pair pipeline for one dr+nurse pair."""

    dr_file_path: Path
    nurse_file_path: Path
    dr_paths: list[Path] = []
    nurse_paths: list[Path] = []
    success: bool
    result: PairClassificationResult | None = None
    error: str | None = None

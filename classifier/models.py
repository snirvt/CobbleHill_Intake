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


class DiagnosisCheck(StrEnum):
    """Whether a note pair carries at least one diagnosis."""

    EXISTS = "EXISTS"
    MISSING = "MISSING"


class CareCheck(StrEnum):
    """Whether a note pair shows the patient needs care (request or medical necessity)."""

    NEEDED = "NEEDED"
    NOT_NEEDED = "NOT_NEEDED"


class Diagnosis(BaseModel):
    """A single diagnosis extracted from a note."""

    name: str
    icd_code: str | None = None


class DiagnosisExtractionResult(BaseModel):
    """All diagnoses extracted from one dr note."""

    file_path: Path
    diagnoses: list[Diagnosis] = []
    reasoning: str | None = None
    # Set when the LLM call or parse failed; diagnoses is then empty, not "none found".
    error: str | None = None


class TreatmentRequestResult(BaseModel):
    """Whether anyone indicates the patient should receive treatment, per a dr note."""

    file_path: Path
    treatment_requested: bool
    reasoning: str | None = None
    # Set when the LLM call or parse failed; treatment_requested is then a placeholder.
    error: str | None = None


class MainDiagnosisResult(BaseModel):
    """Short (1-2 word) primary-diagnosis label for one dr note, used to group cases."""

    file_path: Path
    main_diagnosis: str | None = None  # None = no diagnosis found, or the call failed
    reasoning: str | None = None
    # Set when the LLM call or parse failed.
    error: str | None = None


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
    category: str | None = None  # note-name category prefix, e.g. "hospital" / "peds"


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
    # None when the check did not apply to this pair's category.
    diagnosis_check: DiagnosisCheck | None = None
    dr_diagnoses: list[Diagnosis] = []
    nurse_diagnoses: list[Diagnosis] = []
    care_check: CareCheck | None = None
    care_reasoning: str = ""
    # Short primary-diagnosis label (dr note only) for grouping; not a check.
    main_diagnosis: str | None = None


class PairPipelineResult(BaseModel):
    """Outcome of the pair pipeline for one dr+nurse pair."""

    dr_file_path: Path
    nurse_file_path: Path
    dr_paths: list[Path] = []
    nurse_paths: list[Path] = []
    success: bool
    result: PairClassificationResult | None = None
    error: str | None = None


class PatientAgeResult(BaseModel):
    """Patient age read from a dr note; age_years is None when it cannot be determined."""

    file_path: Path
    age_years: float | None = None
    dob: str | None = None
    date_of_service: str | None = None
    reasoning: str | None = None
    # Set when the LLM call or parse failed; age_years is then None, not "not stated".
    error: str | None = None


class PatientGroup(StrEnum):
    """Which dr-validity rule set applies, decided by age."""

    PEDIATRIC = "PEDIATRIC"
    ADULT = "ADULT"
    UNKNOWN = "UNKNOWN"


class ValidityVerdict(StrEnum):
    """Whether a patient's dr notes satisfy the intake rules for their age group."""

    VALID = "VALID"
    INVALID = "INVALID"
    UNKNOWN_AGE = "UNKNOWN_AGE"


class DrValidityResult(BaseModel):
    """Dr-note validity for one patient (all of the patient's dr notes combined)."""

    patient_folder: Path
    dr_paths: list[Path] = []
    age_years: float | None = None
    patient_group: PatientGroup
    diagnosis_check: DiagnosisCheck
    diagnoses: list[Diagnosis] = []
    # None for adults: medical necessity is not required for them.
    care_check: CareCheck | None = None
    care_reasoning: str = ""
    main_diagnosis: str | None = None
    verdict: ValidityVerdict
    reasons: list[str] = []


class ErrorStage(StrEnum):
    """Where in a run an error happened: folder scan, text extraction, or an LLM step."""

    FOLDER = "folder"
    EXTRACT = "extract"
    AGE = "age"
    DIAGNOSIS = "diagnosis"
    CARE = "care"
    MAIN_DIAGNOSIS = "main_diagnosis"
    PIPELINE = "pipeline"


class ErrorCode(StrEnum):
    """What went wrong."""

    EMPTY_FOLDER = "empty_folder"
    MISSING_REQUIRED_NOTES = "missing_required_notes"
    EMPTY_TEXT = "empty_text"
    LLM_FAILED = "llm_failed"
    PARSE_FAILED = "parse_failed"
    UNEXPECTED = "unexpected"


class RunError(BaseModel):
    """One problem found during a run, reported to the caller instead of being swallowed."""

    stage: ErrorStage
    code: ErrorCode
    message: str
    folder: Path | None = None  # the patient/leaf folder concerned, if any
    file: Path | None = None  # the note concerned, if any


class ClassificationStepError(Exception):
    """Raised by a classifier when one or more of its steps failed after retries.

    Carries every failed step so the pipeline can report them all, rather than a verdict
    computed from fallback defaults.
    """

    def __init__(self, errors: list[RunError]) -> None:
        super().__init__("; ".join(f"{e.stage}: {e.message}" for e in errors))
        self.errors = errors


class DrValidityPipelineResult(BaseModel):
    """Outcome of the dr-validity pipeline for one patient folder."""

    patient_folder: Path
    dr_paths: list[Path] = []
    success: bool
    result: DrValidityResult | None = None
    error: str | None = None
    # Structured errors; may be non-empty even when success is True (non-fatal issues).
    errors: list[RunError] = []

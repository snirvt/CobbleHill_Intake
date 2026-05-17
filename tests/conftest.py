from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from classifier.models import (
    ClassificationResult,
    DocumentMetadata,
    ExaminationData,
    ExtractedFields,
    ExtractedText,
    MedicationData,
    PatientMetadata,
)

SAMPLE_HPI = "Parental Concerns No sleep concerns. Gross/Fine Motor Development normal."
SAMPLE_PLAN = "Follow up in 2 months. Continue current medications. Return if symptoms worsen."
SAMPLE_PREVENTIVE_MEDICINE = (
    "Pediatric Well Visit Counseling: 0-8 Months Infant behavior discussed, "
    "Nutritional adequacy and growth discussed, Safety discussed: car safety seats, "
    "choking hazards discussed, avoid bottle in bed."
)
SAMPLE_ROS = [
    "General NAD.",
    "Respiratory No difficulty breathing.",
    "GI No blood In stool.",
    "GU Normal urine output.",
]
SAMPLE_EXAM_PEDIATRIC = ["GENERAL APPEARANCE: well developed, NAD.", "SKIN: clear."]
SAMPLE_EXAM_GENERAL = "nasal congestion."
SAMPLE_TEXT = f"""\
Test, Patient DOB: 05/30/2025 (8 mo F) Ace No. 399854 DOS: 12/15/2025
Sick Visit Appt
    Patient: Test, Patient                                     Provider: Jimmy Sitt, MD
    Account Number: 399854
    Phone: 347-449-0280
    Address, 380 Henry Street, Brooklyn, NY 11201
Chief Complaints:
         •     Well Visit - 0-11 months
         •     Runny nose, coughing
HPI:
        {SAMPLE_HPI}
    '   ROS:Pediatric
          •   General NAD.
          •   Respiratory No difficulty breathing.
          •   GI No blood In stool.
          •   GU Normal urine output.
E!xamination:
    Pediatdc Exam·
          •   {SAMPLE_EXAM_PEDIATRIC[0]}
          •   {SAMPLE_EXAM_PEDIATRIC[1]}
    General E:xamlnatf:Qo:
    {SAMPLE_EXAM_GENERAL}
Medical History:
         •     Twin B; nsvd; 15 days
         •     hole in heart, follows w cardiology
Surgical History:
         •     brain surgery 09/2025
Hospltalization/Major Diagnostic PrGCedure:
         •     Denies Past Hospitalization
Medications:
    Taking
          •
    Pepsin
    Not-Taking/ PRN
          •   Famotldine 40 MG/5ML suspension Reconstituted o.S ml Orally every 12 hrs
          •   Tylenol Childrens 150 MG/SML Suspension 3 ml Oralty every 4 hours
    Medication List reviewed and reconciled with the patient.
Assessment:
    Assessment.
         •     ENCOUNTER FOR ROUTINE CHILD HEALTH EXAMINATION WITHOUT ABNORMAL FINDINGS - Z00.129 (Primary)
         •     Acute upper respiratory infection, unspecified - J06.9
Procedure Codes:
         •     94760 MEASURE BLOOD OXYGEN LEVEL
Plan:
    {SAMPLE_PLAN}
Preventive Medicine:
    {SAMPLE_PREVENTIVE_MEDICINE}
"""


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    f = tmp_path / "sample.pdf"
    f.write_bytes(b"%PDF-1.4 fake")
    return f


@pytest.fixture
def sample_extracted_text(sample_pdf: Path) -> ExtractedText:
    return ExtractedText(file_path=sample_pdf, text=SAMPLE_TEXT, num_pages=1)


@pytest.fixture
def sample_patient_metadata() -> PatientMetadata:
    return PatientMetadata(
        patient_name="Test, Patient",
        dob="05/30/2025",
        age="8 mo",
        sex="F",
        account_number="399854",
        dos="12/15/2025",
        phone="347-449-0280",
        address="380 Henry Street, Brooklyn, NY 11201",
    )


SAMPLE_COMPLAINTS = ["Well Visit - 0-11 months", "Runny nose, coughing"]
SAMPLE_MEDICAL_HISTORY = ["Twin B; nsvd; 15 days", "hole in heart, follows w cardiology"]
SAMPLE_SURGICAL_HISTORY = ["brain surgery 09/2025"]
SAMPLE_HOSPITALIZATION = ["Denies Past Hospitalization"]
SAMPLE_MEDICATIONS_TAKING = ["Pepsin"]
SAMPLE_MEDICATIONS_NOT_TAKING = [
    "Famotldine 40 MG/5ML suspension Reconstituted o.S ml Orally every 12 hrs",
    "Tylenol Childrens 150 MG/SML Suspension 3 ml Oralty every 4 hours",
]
SAMPLE_MEDICATIONS = MedicationData(
    taking=SAMPLE_MEDICATIONS_TAKING,
    not_taking=SAMPLE_MEDICATIONS_NOT_TAKING,
)
SAMPLE_PROCEDURE_CODES = ["94760 MEASURE BLOOD OXYGEN LEVEL"]
SAMPLE_ASSESSMENT = [
    "ENCOUNTER FOR ROUTINE CHILD HEALTH EXAMINATION WITHOUT ABNORMAL FINDINGS - Z00.129 (Primary)",
    "Acute upper respiratory infection, unspecified - J06.9",
]


SAMPLE_EXAMINATION = ExaminationData(
    pediatric=SAMPLE_EXAM_PEDIATRIC,
    general=SAMPLE_EXAM_GENERAL,
)


@pytest.fixture
def sample_extracted_fields(sample_patient_metadata: PatientMetadata) -> ExtractedFields:
    return ExtractedFields(
        meta=sample_patient_metadata,
        hpi=SAMPLE_HPI,
        examination=SAMPLE_EXAMINATION,
        complaints=SAMPLE_COMPLAINTS,
        medical_history=SAMPLE_MEDICAL_HISTORY,
        surgical_history=SAMPLE_SURGICAL_HISTORY,
        hospitalization=SAMPLE_HOSPITALIZATION,
        assessment=SAMPLE_ASSESSMENT,
        ros=SAMPLE_ROS,
        medications=SAMPLE_MEDICATIONS,
        plan=SAMPLE_PLAN,
        procedure_codes=SAMPLE_PROCEDURE_CODES,
        preventive_medicine=SAMPLE_PREVENTIVE_MEDICINE,
    )


@pytest.fixture
def sample_document_metadata(
    sample_pdf: Path,
    sample_patient_metadata: PatientMetadata,
) -> DocumentMetadata:
    return DocumentMetadata(
        file_path=sample_pdf,
        raw_text=SAMPLE_TEXT,
        meta=sample_patient_metadata,
        hpi=SAMPLE_HPI,
        examination=SAMPLE_EXAMINATION,
        complaints=SAMPLE_COMPLAINTS,
        medical_history=SAMPLE_MEDICAL_HISTORY,
        surgical_history=SAMPLE_SURGICAL_HISTORY,
        hospitalization=SAMPLE_HOSPITALIZATION,
        assessment=SAMPLE_ASSESSMENT,
        ros=SAMPLE_ROS,
        medications=SAMPLE_MEDICATIONS,
        plan=SAMPLE_PLAN,
        procedure_codes=SAMPLE_PROCEDURE_CODES,
        preventive_medicine=SAMPLE_PREVENTIVE_MEDICINE,
    )


@pytest.fixture
def mock_extractor(sample_extracted_text: ExtractedText) -> MagicMock:
    extractor = MagicMock()
    extractor.extract = AsyncMock(return_value=sample_extracted_text)
    return extractor


@pytest.fixture
def mock_meta_extractor(sample_extracted_fields: ExtractedFields) -> MagicMock:
    extractor = MagicMock()
    extractor.extract = MagicMock(return_value=sample_extracted_fields)
    return extractor


@pytest.fixture
def mock_classifier(sample_document_metadata: DocumentMetadata) -> MagicMock:
    clf = MagicMock()
    clf.classify = AsyncMock(
        return_value=ClassificationResult(
            file_path=sample_document_metadata.file_path,
            category="TEST",
            metadata=sample_document_metadata,
        )
    )
    return clf


@pytest.fixture
def mock_router(mock_extractor: MagicMock) -> MagicMock:
    router = MagicMock()
    router.route = MagicMock(return_value=mock_extractor)
    return router

import pytest

from classifier.metadata.progress_note import ProgressNoteMetadataExtractor
from classifier.models import ExaminationData, ExtractedFields
from tests.conftest import (
    SAMPLE_ASSESSMENT,
    SAMPLE_EXAM_GENERAL,
    SAMPLE_EXAM_PEDIATRIC,
    SAMPLE_HPI,
    SAMPLE_MEDICATIONS_NOT_TAKING,
    SAMPLE_MEDICATIONS_TAKING,
    SAMPLE_PLAN,
    SAMPLE_PREVENTIVE_MEDICINE,
    SAMPLE_PROCEDURE_CODES,
    SAMPLE_ROS,
    SAMPLE_TEXT,
)


@pytest.fixture
def extractor() -> ProgressNoteMetadataExtractor:
    return ProgressNoteMetadataExtractor()


def test_extract_returns_extracted_fields(extractor: ProgressNoteMetadataExtractor) -> None:
    result = extractor.extract(SAMPLE_TEXT)
    assert isinstance(result, ExtractedFields)


def test_extracts_patient_name(extractor: ProgressNoteMetadataExtractor) -> None:
    result = extractor.extract(SAMPLE_TEXT)
    assert result.meta.patient_name == "Test, Patient"


def test_extracts_dob(extractor: ProgressNoteMetadataExtractor) -> None:
    result = extractor.extract(SAMPLE_TEXT)
    assert result.meta.dob == "05/30/2025"


def test_extracts_age_from_inline_header(extractor: ProgressNoteMetadataExtractor) -> None:
    result = extractor.extract(SAMPLE_TEXT)
    assert result.meta.age == "8 mo"


def test_extracts_sex_from_inline_header(extractor: ProgressNoteMetadataExtractor) -> None:
    result = extractor.extract(SAMPLE_TEXT)
    assert result.meta.sex == "F"


def test_extracts_account_number(extractor: ProgressNoteMetadataExtractor) -> None:
    result = extractor.extract(SAMPLE_TEXT)
    assert result.meta.account_number == "399854"


def test_extracts_dos(extractor: ProgressNoteMetadataExtractor) -> None:
    result = extractor.extract(SAMPLE_TEXT)
    assert result.meta.dos == "12/15/2025"


def test_extracts_phone(extractor: ProgressNoteMetadataExtractor) -> None:
    result = extractor.extract(SAMPLE_TEXT)
    assert result.meta.phone == "347-449-0280"


def test_extracts_address(extractor: ProgressNoteMetadataExtractor) -> None:
    result = extractor.extract(SAMPLE_TEXT)
    assert result.meta.address == "380 Henry Street, Brooklyn, NY 11201"


def test_extracts_hpi(extractor: ProgressNoteMetadataExtractor) -> None:
    result = extractor.extract(SAMPLE_TEXT)
    assert result.hpi == SAMPLE_HPI


def test_hpi_stops_at_ros(extractor: ProgressNoteMetadataExtractor) -> None:
    text = "HPI:\n        First line of HPI\n        Second line\n    '   ROS:Pediatric\n"
    result = extractor.extract(text)
    assert result.hpi == "First line of HPI Second line"


def test_hpi_stops_at_section_header(extractor: ProgressNoteMetadataExtractor) -> None:
    text = "HPI:\n        HPI content here\nMedical History:\n         •     item\n"
    result = extractor.extract(text)
    assert result.hpi == "HPI content here"


def test_hpi_joins_multiline(extractor: ProgressNoteMetadataExtractor) -> None:
    text = "HPI:\n        Line one text\n        Line two text\n        Line three\nROS:\n"
    result = extractor.extract(text)
    assert result.hpi == "Line one text Line two text Line three"


def test_hpi_none_when_missing(extractor: ProgressNoteMetadataExtractor) -> None:
    result = extractor.extract("no structured data here")
    assert result.hpi is None


def test_extracts_examination_returns_examination_data(
    extractor: ProgressNoteMetadataExtractor,
) -> None:
    result = extractor.extract(SAMPLE_TEXT)
    assert isinstance(result.examination, ExaminationData)


def test_extracts_examination_pediatric_bullets(
    extractor: ProgressNoteMetadataExtractor,
) -> None:
    result = extractor.extract(SAMPLE_TEXT)
    assert result.examination is not None
    assert result.examination.pediatric == SAMPLE_EXAM_PEDIATRIC


def test_extracts_examination_general_text(
    extractor: ProgressNoteMetadataExtractor,
) -> None:
    result = extractor.extract(SAMPLE_TEXT)
    assert result.examination is not None
    assert result.examination.general == SAMPLE_EXAM_GENERAL


def test_examination_pediatric_handles_ocr_dropped_letters(
    extractor: ProgressNoteMetadataExtractor,
) -> None:
    # OCR drops "ri" from "Pediatric" → "Pediatdc"; also uses · instead of :
    text = "Examination:\n    Pediatdc Exam·\n          •   HEENT: normal.\nAssessment:\n"
    result = extractor.extract(text)
    assert result.examination is not None
    assert result.examination.pediatric == ["HEENT: normal."]


def test_examination_general_handles_ocr_garbled_header(
    extractor: ProgressNoteMetadataExtractor,
) -> None:
    text = (
        "Examination:\n"
        "    Pediatric Exam:\n"
        "          •   HEENT: normal.\n"
        "    General E:xamlnatf:Qo:\n"
        "    nasal congestion.\n"
        "Assessment:\n"
    )
    result = extractor.extract(text)
    assert result.examination is not None
    assert result.examination.general == "nasal congestion."


def test_examination_none_when_missing(extractor: ProgressNoteMetadataExtractor) -> None:
    result = extractor.extract("no structured data here")
    assert result.examination is None


def test_extracts_chief_complaints(extractor: ProgressNoteMetadataExtractor) -> None:
    result = extractor.extract(SAMPLE_TEXT)
    assert result.complaints == ["Well Visit - 0-11 months", "Runny nose, coughing"]


def test_complaints_strips_trailing_backslash(extractor: ProgressNoteMetadataExtractor) -> None:
    text = "Chief Complaints:\n         •     Previsit item\\\n         •     Normal item\n"
    result = extractor.extract(text)
    assert result.complaints == ["Previsit item", "Normal item"]


def test_complaints_empty_when_section_missing(extractor: ProgressNoteMetadataExtractor) -> None:
    result = extractor.extract("no structured data here")
    assert result.complaints == []


def test_extracts_medical_history(extractor: ProgressNoteMetadataExtractor) -> None:
    result = extractor.extract(SAMPLE_TEXT)
    assert result.medical_history == ["Twin B; nsvd; 15 days", "hole in heart, follows w cardiology"]


def test_medical_history_empty_when_section_missing(extractor: ProgressNoteMetadataExtractor) -> None:
    result = extractor.extract("Chief Complaints:\n         •     Cough\n")
    assert result.medical_history == []


def test_medical_history_independent_of_complaints(extractor: ProgressNoteMetadataExtractor) -> None:
    text = "Medical History:\n         •     Asthma\n         •     Diabetes\n"
    result = extractor.extract(text)
    assert result.medical_history == ["Asthma", "Diabetes"]
    assert result.complaints == []


def test_medical_history_joins_wrapped_lines(extractor: ProgressNoteMetadataExtractor) -> None:
    text = (
        "Medical History:\n"
        "         •     NICU for meningitis, discharged after 14 days, follows w\n"
        "         developmental ped and neurology.\n"
        "         •     hole in heart follows w cardiology\n"
        "Surgical History:\n"
    )
    result = extractor.extract(text)
    assert result.medical_history == [
        "NICU for meningitis, discharged after 14 days, follows w developmental ped and neurology.",
        "hole in heart follows w cardiology",
    ]


def test_extracts_surgical_history(extractor: ProgressNoteMetadataExtractor) -> None:
    result = extractor.extract(SAMPLE_TEXT)
    assert result.surgical_history == ["brain surgery 09/2025"]


def test_extracts_hospitalization(extractor: ProgressNoteMetadataExtractor) -> None:
    result = extractor.extract(SAMPLE_TEXT)
    assert result.hospitalization == ["Denies Past Hospitalization"]


def test_hospitalization_matches_ocr_garbled_header(
    extractor: ProgressNoteMetadataExtractor,
) -> None:
    text = "Hospltalization/Major Diagnostic PrGCedure:\n         •     NICU stay 14 days\n"
    result = extractor.extract(text)
    assert result.hospitalization == ["NICU stay 14 days"]


def test_hospitalization_matches_clean_header(
    extractor: ProgressNoteMetadataExtractor,
) -> None:
    text = "Hospitalization/Major Diagnostic Procedure:\n         •     Appendectomy 2020\n"
    result = extractor.extract(text)
    assert result.hospitalization == ["Appendectomy 2020"]


def test_complaints_joins_wrapped_lines(extractor: ProgressNoteMetadataExtractor) -> None:
    text = (
        "Chief Complaints:\n"
        "         •     Well Visit - 0-11 months with extended notes that\n"
        "         wrap to the next line\n"
        "         •     Runny nose\n"
        "HPI:\n"
    )
    result = extractor.extract(text)
    assert result.complaints == [
        "Well Visit - 0-11 months with extended notes that wrap to the next line",
        "Runny nose",
    ]


def test_returns_none_for_missing_meta_fields(extractor: ProgressNoteMetadataExtractor) -> None:
    result = extractor.extract("no structured data here")
    assert result.meta.patient_name is None
    assert result.meta.dob is None
    assert result.meta.age is None
    assert result.meta.sex is None
    assert result.meta.account_number is None
    assert result.meta.dos is None
    assert result.meta.phone is None
    assert result.meta.address is None


def test_extracts_sex_from_label_fallback(extractor: ProgressNoteMetadataExtractor) -> None:
    text = "Patient: Doe, Jane\nDOB: 01/01/2020\nDOS: 03/15/2025\nsex: Female\nPhone: 555-0000\nAddress, 1 Main St"
    result = extractor.extract(text)
    assert result.meta.sex == "Female"
    assert result.meta.age is None


def test_extracts_assessment(extractor: ProgressNoteMetadataExtractor) -> None:
    result = extractor.extract(SAMPLE_TEXT)
    assert result.assessment == SAMPLE_ASSESSMENT


def test_assessment_empty_when_section_missing(extractor: ProgressNoteMetadataExtractor) -> None:
    result = extractor.extract("no structured data here")
    assert result.assessment == []


def test_assessment_skips_repeated_label_line(extractor: ProgressNoteMetadataExtractor) -> None:
    text = (
        "Assessment:\n"
        "    Assessment.\n"
        "         •     Fever - R50.9\n"
        "         •     Cough - R05\n"
    )
    result = extractor.extract(text)
    assert result.assessment == ["Fever - R50.9", "Cough - R05"]


def test_assessment_independent_of_hospitalization(extractor: ProgressNoteMetadataExtractor) -> None:
    text = "Assessment:\n         •     Otitis media - H66.9\n"
    result = extractor.extract(text)
    assert result.assessment == ["Otitis media - H66.9"]
    assert result.hospitalization == []


def test_extracts_ros(extractor: ProgressNoteMetadataExtractor) -> None:
    result = extractor.extract(SAMPLE_TEXT)
    assert result.ros == SAMPLE_ROS


def test_ros_empty_when_section_missing(extractor: ProgressNoteMetadataExtractor) -> None:
    result = extractor.extract("no structured data here")
    assert result.ros == []


def test_ros_stops_before_examination(extractor: ProgressNoteMetadataExtractor) -> None:
    text = (
        "ROS:Pediatric\n"
        "          •   General NAD.\n"
        "          •   Respiratory clear.\n"
        "E!xamination:\n"
        "    Pediatric Exam:\n"
        "          •   HEENT: normal.\n"
    )
    result = extractor.extract(text)
    assert result.ros == ["General NAD.", "Respiratory clear."]
    assert result.examination is not None
    assert result.examination.pediatric == ["HEENT: normal."]


def test_ros_independent_of_hpi(extractor: ProgressNoteMetadataExtractor) -> None:
    text = "ROS:Pediatric\n          •   GI No vomiting.\n"
    result = extractor.extract(text)
    assert result.ros == ["GI No vomiting."]
    assert result.hpi is None


def test_extracts_medications_taking(extractor: ProgressNoteMetadataExtractor) -> None:
    result = extractor.extract(SAMPLE_TEXT)
    assert result.medications is not None
    assert result.medications.taking == SAMPLE_MEDICATIONS_TAKING


def test_extracts_medications_not_taking(extractor: ProgressNoteMetadataExtractor) -> None:
    result = extractor.extract(SAMPLE_TEXT)
    assert result.medications is not None
    assert result.medications.not_taking == SAMPLE_MEDICATIONS_NOT_TAKING


def test_medications_none_when_section_missing(extractor: ProgressNoteMetadataExtractor) -> None:
    result = extractor.extract("no structured data here")
    assert result.medications is None


def test_medications_bare_bullet_item_on_next_line(extractor: ProgressNoteMetadataExtractor) -> None:
    text = "Medications:\n    Taking\n          •\n    Amoxicillin\n"
    result = extractor.extract(text)
    assert result.medications is not None
    assert result.medications.taking == ["Amoxicillin"]


def test_medications_stops_at_reviewed_sentence(extractor: ProgressNoteMetadataExtractor) -> None:
    text = (
        "Medications:\n"
        "    Not-Taking/ PRN\n"
        "          •   Ibuprofen 100mg\n"
        "    Medication List reviewed and reconciled with the patient.\n"
        "Allergies:\n"
        "          •   Penicillin\n"
    )
    result = extractor.extract(text)
    assert result.medications is not None
    assert result.medications.not_taking == ["Ibuprofen 100mg"]


def test_medications_empty_sub_lists_when_no_bullets(extractor: ProgressNoteMetadataExtractor) -> None:
    text = "Medications:\n    Taking\n    Not-Taking/ PRN\n    Medication List reviewed.\nAllergies:\n"
    result = extractor.extract(text)
    assert result.medications is not None
    assert result.medications.taking == []
    assert result.medications.not_taking == []


def test_extracts_plan(extractor: ProgressNoteMetadataExtractor) -> None:
    result = extractor.extract(SAMPLE_TEXT)
    assert result.plan == SAMPLE_PLAN


def test_plan_none_when_missing(extractor: ProgressNoteMetadataExtractor) -> None:
    result = extractor.extract("no structured data here")
    assert result.plan is None


def test_plan_joins_multiline(extractor: ProgressNoteMetadataExtractor) -> None:
    text = "Plan:\n    Follow up in 2 months.\n    Return if symptoms worsen.\nAllergies:\n"
    result = extractor.extract(text)
    assert result.plan == "Follow up in 2 months. Return if symptoms worsen."


def test_plan_stops_at_next_section(extractor: ProgressNoteMetadataExtractor) -> None:
    text = "Plan:\n    Rest and hydration.\nAllergies:\n    •   Penicillin\n"
    result = extractor.extract(text)
    assert result.plan == "Rest and hydration."


def test_plan_stops_at_fax_footer_clean(extractor: ProgressNoteMetadataExtractor) -> None:
    text = (
        "Plan:\n"
        "    1. Follow up in 2 months.\n"
        "    2. Return if fever develops.\n"
        "Document: Signed Dr. Smith 02/18/26    Printed: 02-18-2026 11:08:25 Page 5 of 6\n"
        "2/18/2026 FROM: Fax ODA TO: 17188557082\n"
    )
    result = extractor.extract(text)
    assert result.plan == "1. Follow up in 2 months. 2. Return if fever develops."


def test_plan_stops_at_ocr_garbled_fax_footer(extractor: ProgressNoteMetadataExtractor) -> None:
    # OCR drops "D" from "Document" → ")ocument: Signed"
    text = (
        "Plan:\n"
        "    Return in 2-3 days.\n"
        ")ocument: Signed Yourchoice 02/18/26    Printed: 02-18-2026 11:08:25 Page 5 of 6\n"
        "2/18/2026 11:09 AM FROM: Fax ODA - Wallabout TO: 17188557082 PAGE: 007 OF 007\n"
        "•atient: Test, Patient DOB: May 30, 2025\n"
    )
    result = extractor.extract(text)
    assert result.plan == "Return in 2-3 days."


def test_extracts_procedure_codes(extractor: ProgressNoteMetadataExtractor) -> None:
    result = extractor.extract(SAMPLE_TEXT)
    assert result.procedure_codes == SAMPLE_PROCEDURE_CODES


def test_procedure_codes_empty_when_section_missing(extractor: ProgressNoteMetadataExtractor) -> None:
    result = extractor.extract("no structured data here")
    assert result.procedure_codes == []


def test_procedure_codes_multiple_bullets(extractor: ProgressNoteMetadataExtractor) -> None:
    text = (
        "Procedure Codes:\n"
        "         •     99213 OFFICE VISIT ESTABLISHED\n"
        "         •     94760 MEASURE BLOOD OXYGEN LEVEL\n"
    )
    result = extractor.extract(text)
    assert result.procedure_codes == ["99213 OFFICE VISIT ESTABLISHED", "94760 MEASURE BLOOD OXYGEN LEVEL"]


def test_extracts_preventive_medicine(extractor: ProgressNoteMetadataExtractor) -> None:
    result = extractor.extract(SAMPLE_TEXT)
    assert result.preventive_medicine == SAMPLE_PREVENTIVE_MEDICINE


def test_preventive_medicine_none_when_missing(extractor: ProgressNoteMetadataExtractor) -> None:
    result = extractor.extract("no structured data here")
    assert result.preventive_medicine is None


def test_preventive_medicine_joins_multiline(extractor: ProgressNoteMetadataExtractor) -> None:
    text = (
        "Preventive Medicine:\n"
        "    Infant behavior discussed.\n"
        "    Safety discussed: car seats.\n"
        "Allergies:\n"
    )
    result = extractor.extract(text)
    assert result.preventive_medicine == "Infant behavior discussed. Safety discussed: car seats."


def test_preventive_medicine_stops_at_next_section(extractor: ProgressNoteMetadataExtractor) -> None:
    text = "Preventive Medicine:\n    Growth discussed.\nProcedure Codes:\n    •   99213 OFFICE VISIT\n"
    result = extractor.extract(text)
    assert result.preventive_medicine == "Growth discussed."


def test_preventive_medicine_stops_at_follow_up(extractor: ProgressNoteMetadataExtractor) -> None:
    # "Follow Up: At 9 Months,prn" has content after colon — doesn't match _SECTION_HEADER
    text = (
        "Preventive Medicine:\n"
        "    Safety discussed: car seats, avoid bottle in bed.\n"
        "Follow Up: At 9 Months,prn\n"
        "care Plan: Problem:\n"
    )
    result = extractor.extract(text)
    assert result.preventive_medicine == "Safety discussed: car seats, avoid bottle in bed."


def test_preventive_medicine_stops_at_ocr_lowercase_care_plan(
    extractor: ProgressNoteMetadataExtractor,
) -> None:
    # OCR lowercases "C" → "care Plan:" which doesn't match _SECTION_HEADER
    text = (
        "Preventive Medicine:\n"
        "    Nutrition discussed.\n"
        "care Plan: Problem:\n"
    )
    result = extractor.extract(text)
    assert result.preventive_medicine == "Nutrition discussed."


def test_strip_fax_noise_removes_transmission_lines(extractor: ProgressNoteMetadataExtractor) -> None:
    text = (
        "Plan:\n"
        "    Follow up next week.\n"
        "2/18/2026 11:09 AM FROM: Fax ODA TO: 17188557082 PAGE: 007 OF 007\n"
        "eCW (Nibbs, Glorybea) Production Environment\n"
    )
    result = extractor.extract(text)
    assert result.plan == "Follow up next week."
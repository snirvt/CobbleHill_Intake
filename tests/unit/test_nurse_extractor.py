import pytest

from classifier.metadata.nurse_visit import NurseVisitExtractor
from classifier.models import NurseVisitFields

SAMPLE_NURSE_TEXT = """\
4/1/26, 12:43 PM            Encounter - Home Visit Date of service: 02/19/26 Patient: fname lname DOB: 02/14/2025 PRN: 1234567A

PATIENT                                FACILITY                              ENCOUNTER
fname lname                            NYC House Call Medical                Home Visit
DOB         02/14/2025                 T  (918) 453-1833                     NOTE TYPE      SOAP Note
AGE         2 mos                      F  (7819) 859-9409                    SEEN BY        Kelly Sokolinsky
SEX         Female                     70 W Brighton Ave                     DATE           04/17/2026
PRN         1234567A                   Brooklyn, NY 11224                    AGE AT DOS     0 mos

Chief complaint
Feeding difficulties, excessive crying, gas, and irritability

 FIRST NAME             fname          SEX                   Female
 LAST NAME              lname          DATE OF BIRTH         02/14/2025
 PRN                    1234567A
 ADDRESS LINE 1         2334 AVENUE Z
 CITY                   Brooklyn
 STATE                  NY
 ZIP CODE               11229
 MOBILE PHONE          (828) 574-78889

Diagnoses

 Current
 (R63.30) Feeding difficulties, unspecified
 (K21.9) Gastro-esophageal reflux disease without esophagitis
 (R10.83) Colic

Current Medications (as of 04/01/2026)

Was medication reconciliation completed?
Yes, reconciliation performed

Active                                  SIG     START/STOP   ASSOCIATED DX

No active medications recorded

Historical                              SIG     START/STOP   ASSOCIATED DX

No historical medications recorded

Subjective

43-day-old female presents with caregiver-reported concerns of persistent colic, feeding intolerance, gas, constipation, and irritability.

Objective

PHYSICAL EXAM

General: Awake, intermittently irritable, consolable, small for age
Vital Signs: Temperature 98.4F within normal limits

GROWTH PARAMETERS

Weight: 2.7 kg

Assessment

Feeding difficulties in newborn - R63.3
Gastroesophageal reflux disease - K21.9
Infantile colic - R10.83

Plan

Continue breastfeeding/formula every 2-3 hours. Smaller, more frequent feeds recommended.

Care plan

CHHA Skilled Nursing for feeding monitoring, weight checks, hydration assessment.
"""


@pytest.fixture
def extractor() -> NurseVisitExtractor:
    return NurseVisitExtractor()


@pytest.fixture
def fields(extractor: NurseVisitExtractor) -> NurseVisitFields:
    return extractor.extract(SAMPLE_NURSE_TEXT)


def test_nurse_extractor_patient_name(fields: NurseVisitFields) -> None:
    assert fields.meta.patient_name == "fname lname"


def test_nurse_extractor_dob(fields: NurseVisitFields) -> None:
    assert fields.meta.dob == "02/14/2025"


def test_nurse_extractor_dos(fields: NurseVisitFields) -> None:
    assert fields.meta.dos == "02/19/26"


def test_nurse_extractor_sex(fields: NurseVisitFields) -> None:
    assert fields.meta.sex is not None
    assert fields.meta.sex.lower() == "female"


def test_nurse_extractor_prn(fields: NurseVisitFields) -> None:
    assert fields.meta.prn == "1234567A"


def test_nurse_extractor_seen_by(fields: NurseVisitFields) -> None:
    assert fields.meta.seen_by is not None
    assert "Kelly Sokolinsky" in fields.meta.seen_by


def test_nurse_extractor_address(fields: NurseVisitFields) -> None:
    assert fields.meta.address is not None
    assert "2334 AVENUE Z" in fields.meta.address
    assert "Brooklyn" in fields.meta.address


def test_nurse_extractor_phone(fields: NurseVisitFields) -> None:
    assert fields.meta.phone is not None
    assert "828" in fields.meta.phone


def test_nurse_extractor_chief_complaint(fields: NurseVisitFields) -> None:
    assert fields.chief_complaint is not None
    assert "Feeding difficulties" in fields.chief_complaint


def test_nurse_extractor_diagnoses(fields: NurseVisitFields) -> None:
    assert len(fields.diagnoses) == 3
    assert any("(R63.30)" in d for d in fields.diagnoses)
    assert any("(K21.9)" in d for d in fields.diagnoses)


def test_nurse_extractor_subjective(fields: NurseVisitFields) -> None:
    assert fields.subjective is not None
    assert "colic" in fields.subjective.lower()


def test_nurse_extractor_assessment(fields: NurseVisitFields) -> None:
    assert len(fields.assessment) >= 1
    assert any("R63.3" in a for a in fields.assessment)


def test_nurse_extractor_plan(fields: NurseVisitFields) -> None:
    assert fields.plan is not None
    assert "breastfeeding" in fields.plan.lower()


def test_nurse_extractor_care_plan(fields: NurseVisitFields) -> None:
    assert fields.care_plan is not None
    assert "CHHA" in fields.care_plan


def test_nurse_extractor_no_active_medications(fields: NurseVisitFields) -> None:
    assert fields.medications_active == []


def test_nurse_extractor_plain_text_fallback_uses_subjective() -> None:
    raw = "Patient is 7 weeks old and weighs 7.5 pounds, gaining weight properly."
    extractor = NurseVisitExtractor()
    fields = extractor.extract(raw)
    assert fields.subjective == raw
    assert fields.meta.patient_name is None
    assert fields.diagnoses == []
    assert fields.assessment == []

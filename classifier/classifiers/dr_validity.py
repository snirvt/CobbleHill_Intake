import asyncio
import logging
from pathlib import Path

from classifier.models import (
    CareCheck,
    DiagnosisCheck,
    DocumentMetadata,
    DrValidityResult,
    PatientGroup,
    ValidityVerdict,
)
from classifier.protocols import (
    DiagnosisExtractor,
    MainDiagnosisExtractor,
    PatientAgeExtractor,
    TreatmentRequestExtractor,
)
from config.settings import settings

logger = logging.getLogger(__name__)

_REASON_NO_DIAGNOSIS = "no diagnosis"
_REASON_NO_NECESSITY = "no medical necessity"
_REASON_NO_NECESSITY_IF_PEDS = "no medical necessity (required if under 18)"
_REASON_UNKNOWN_AGE = "age not found"


class DrNoteValidityClassifier:
    """Checks a patient's combined dr notes against the intake rules for their age.

    Pediatric (under ``settings.pediatric_age_threshold``): needs a diagnosis AND medical
    necessity. Adult: needs a diagnosis. Every patient also gets a short main-diagnosis
    label for grouping. The four LLM steps run concurrently.
    """

    def __init__(
        self,
        age_extractor: PatientAgeExtractor,
        diagnosis_extractor: DiagnosisExtractor,
        care_extractor: TreatmentRequestExtractor,
        main_diagnosis_extractor: MainDiagnosisExtractor,
    ) -> None:
        self._age_extractor = age_extractor
        self._diagnosis_extractor = diagnosis_extractor
        self._care_extractor = care_extractor
        self._main_diagnosis_extractor = main_diagnosis_extractor

    async def classify(
        self, metadata: DocumentMetadata, dr_paths: list[Path]
    ) -> DrValidityResult:
        """Run all checks on the combined dr-note text and return the patient's verdict.

        ``metadata.file_path`` is the patient's first dr note; its folder is the patient folder.
        """
        age, diagnoses, care, main = await asyncio.gather(
            self._age_extractor.extract_age(metadata),
            self._diagnosis_extractor.extract_diagnoses(metadata),
            self._care_extractor.extract_treatment_request(metadata),
            self._main_diagnosis_extractor.extract_main_diagnosis(metadata),
        )
        group = self._patient_group(age.age_years)
        diagnosis_check = DiagnosisCheck.EXISTS if diagnoses.diagnoses else DiagnosisCheck.MISSING
        care_check = CareCheck.NEEDED if care.treatment_requested else CareCheck.NOT_NEEDED
        verdict, reasons = self._verdict(group, diagnosis_check, care_check)
        logger.debug(
            "Dr validity %s for %s (group=%s, reasons=%s)",
            verdict,
            metadata.file_path,
            group,
            reasons,
        )
        is_adult = group is PatientGroup.ADULT
        return DrValidityResult(
            patient_folder=metadata.file_path.parent,
            dr_paths=dr_paths,
            age_years=age.age_years,
            patient_group=group,
            diagnosis_check=diagnosis_check,
            diagnoses=diagnoses.diagnoses,
            care_check=None if is_adult else care_check,
            care_reasoning="" if is_adult else care.reasoning or "",
            main_diagnosis=main.main_diagnosis,
            verdict=verdict,
            reasons=reasons,
        )

    @staticmethod
    def _patient_group(age_years: float | None) -> PatientGroup:
        """Map an age to its rule set; None means the age could not be determined."""
        if age_years is None:
            return PatientGroup.UNKNOWN
        if age_years < settings.pediatric_age_threshold:
            return PatientGroup.PEDIATRIC
        return PatientGroup.ADULT

    @staticmethod
    def _verdict(
        group: PatientGroup, diagnosis_check: DiagnosisCheck, care_check: CareCheck
    ) -> tuple[ValidityVerdict, list[str]]:
        """Apply the age group's rules; return the verdict and what is missing."""
        reasons: list[str] = []
        if diagnosis_check is DiagnosisCheck.MISSING:
            reasons.append(_REASON_NO_DIAGNOSIS)
        no_necessity = care_check is CareCheck.NOT_NEEDED

        if group is PatientGroup.PEDIATRIC:
            if no_necessity:
                reasons.append(_REASON_NO_NECESSITY)
        elif group is PatientGroup.UNKNOWN:
            reasons.append(_REASON_UNKNOWN_AGE)
            if no_necessity:
                reasons.append(_REASON_NO_NECESSITY_IF_PEDS)

        if diagnosis_check is DiagnosisCheck.MISSING or (
            group is PatientGroup.PEDIATRIC and no_necessity
        ):
            return ValidityVerdict.INVALID, reasons
        if group is PatientGroup.UNKNOWN:
            return ValidityVerdict.UNKNOWN_AGE, reasons
        return ValidityVerdict.VALID, reasons

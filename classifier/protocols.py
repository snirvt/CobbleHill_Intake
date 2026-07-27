from pathlib import Path
from typing import Protocol, runtime_checkable

from classifier.models import (
    ClassificationResult,
    DiagnosisExtractionResult,
    DocumentMetadata,
    ExtractedFields,
    ExtractedText,
    NurseVisitFields,
    PairClassificationResult,
    PairDocumentMetadata,
    TreatmentRequestResult,
)


@runtime_checkable
class TaskClassifier(Protocol):
    """Single-purpose classifier that returns a category string for one task."""

    task_name: str

    async def run(self, metadata: DocumentMetadata) -> str: ...


@runtime_checkable
class ContentExtractor(Protocol):
    """Extracts raw text from a file."""

    async def extract(self, file_path: Path) -> ExtractedText: ...


@runtime_checkable
class NoteExtractor(Protocol):
    """Parses structured fields from extracted document text."""

    def extract(self, text: str) -> ExtractedFields: ...


@runtime_checkable
class Classifier(Protocol):
    """Classifies a document given its metadata."""

    async def classify(self, metadata: DocumentMetadata) -> ClassificationResult: ...


@runtime_checkable
class FileRouter(Protocol):
    """Resolves a file path to the appropriate ContentExtractor."""

    def route(self, file_path: Path) -> ContentExtractor: ...


@runtime_checkable
class NurseNoteExtractor(Protocol):
    """Parses structured fields from a nurse visit note."""

    def extract(self, text: str) -> NurseVisitFields: ...


@runtime_checkable
class PairClassifier(Protocol):
    """Classifies a dr+nurse document pair."""

    async def classify_pair(self, pair: PairDocumentMetadata) -> PairClassificationResult: ...


@runtime_checkable
class DiagnosisExtractor(Protocol):
    """Extracts diagnoses (name + optional ICD code) from a dr note."""

    async def extract_diagnoses(
        self, metadata: DocumentMetadata
    ) -> DiagnosisExtractionResult: ...


@runtime_checkable
class TreatmentRequestExtractor(Protocol):
    """Determines whether the patient explicitly requested treatment in a dr note."""

    async def extract_treatment_request(
        self, metadata: DocumentMetadata
    ) -> TreatmentRequestResult: ...

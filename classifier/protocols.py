from pathlib import Path
from typing import Protocol, runtime_checkable

from classifier.models import (
    ClassificationResult,
    DocumentMetadata,
    ExtractedFields,
    ExtractedText,
)


@runtime_checkable
class LLMProvider(Protocol):
    """Sends a prompt to an LLM and returns the raw text response."""

    async def complete(self, prompt: str) -> str: ...


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
class MetadataExtractor(Protocol):
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

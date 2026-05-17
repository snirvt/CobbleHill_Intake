from classifier.models import ClassificationResult, DocumentMetadata


class DummyClassifier:
    """Stub classifier — always returns category 'TEST'.

    Replace with real classifier implementations when ready.
    """

    async def classify(self, metadata: DocumentMetadata) -> ClassificationResult:
        """Return a fixed TEST classification for any document."""
        return ClassificationResult(
            file_path=metadata.file_path,
            category="TEST",
            metadata=metadata,
        )

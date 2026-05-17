import asyncio
import logging

from classifier.models import ClassificationResult, DocumentMetadata
from classifier.protocols import TaskClassifier

logger = logging.getLogger(__name__)


class LLMClassifier:
    """Orchestrates multiple TaskClassifiers in parallel and aggregates results."""

    def __init__(self, tasks: list[TaskClassifier]) -> None:
        self._tasks = tasks

    async def classify(self, metadata: DocumentMetadata) -> ClassificationResult:
        """Run all tasks concurrently; primary category comes from the first task."""
        results: list[str] = await asyncio.gather(
            *[task.run(metadata) for task in self._tasks]
        )
        task_results = {
            task.task_name: result
            for task, result in zip(self._tasks, results)
        }
        primary = results[0] if results else "UNKNOWN"
        logger.debug("Classification complete file=%s results=%s", metadata.file_path, task_results)
        return ClassificationResult(
            file_path=metadata.file_path,
            category=primary,
            task_results=task_results,
            metadata=metadata,
        )

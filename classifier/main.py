import asyncio
import os

from classifier.classifiers.doctor_visit_needed import DoctorVisitNeededClassifier
from classifier.classifiers.llm_classifier import LLMClassifier
from classifier.extractors.pdf import PdfExtractor
from classifier.metadata.progress_note import ProgressNoteMetadataExtractor
from classifier.pipeline import Pipeline
from classifier.providers.ollama import OllamaProvider
from classifier.routing import DefaultFileRouter
from config.settings import settings

_TASK_REGISTRY = {
    "doctor_visit_needed": DoctorVisitNeededClassifier,
}


def build_pipeline() -> Pipeline:
    """Wire up all components and return a ready-to-use Pipeline."""
    if settings.node_bin_path not in os.environ.get("PATH", ""):
        os.environ["PATH"] = settings.node_bin_path + ":" + os.environ["PATH"]

    extractor = PdfExtractor()
    router = DefaultFileRouter({"pdf": extractor})
    meta_extractor = ProgressNoteMetadataExtractor()

    llm_semaphore = asyncio.Semaphore(settings.max_concurrent_llm_calls)
    llm = OllamaProvider(
        url=settings.ollama_url,
        model=settings.ollama_model,
        semaphore=llm_semaphore,
    )
    tasks = [
        _TASK_REGISTRY[name](llm)
        for name in settings.classifier_tasks
        if name in _TASK_REGISTRY
    ]
    classifier = LLMClassifier(tasks=tasks)

    return Pipeline(router=router, meta_extractor=meta_extractor, classifier=classifier)

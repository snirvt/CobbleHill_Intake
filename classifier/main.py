import asyncio
import os
from functools import cache

from langchain_core.language_models import BaseChatModel

from classifier.classifiers.doctor_visit_needed import DoctorVisitNeededClassifier
from classifier.classifiers.dr_nurse_match import DrNurseMatchClassifier
from classifier.classifiers.llm_classifier import LLMClassifier
from classifier.extractors.easyocr_extractor import (
    EasyOcrExtractor,
    OcrReader,
    create_easyocr_reader,
)
from classifier.extractors.pdf import PdfExtractor
from classifier.extractors.plaintext import PlaintextExtractor
from classifier.metadata.nurse_visit import NurseVisitExtractor
from classifier.metadata.progress_note import ProgressNoteExtractor
from classifier.pair_pipeline import PairPipeline
from classifier.pipeline import Pipeline
from classifier.providers.ollama import create_ollama_chat_model
from classifier.routing import DefaultFileRouter
from config.settings import settings

_TASK_REGISTRY = {
    "doctor_visit_needed": DoctorVisitNeededClassifier,
}


def _make_env() -> None:
    if settings.node_bin_path not in os.environ.get("PATH", ""):
        os.environ["PATH"] = settings.node_bin_path + ":" + os.environ["PATH"]


@cache
def _make_ocr_reader() -> OcrReader:
    """Build the easyocr Reader once and reuse it (model load is expensive)."""
    return create_easyocr_reader(
        languages=settings.ocr_languages,
        gpu=settings.ocr_gpu,
        model_storage_directory=settings.ocr_model_dir,
        download_enabled=settings.ocr_download,
    )


def _make_llm() -> BaseChatModel:
    semaphore = asyncio.Semaphore(settings.max_concurrent_llm_calls)
    return create_ollama_chat_model(
        url=settings.ollama_url,
        model=settings.ollama_model,
        semaphore=semaphore,
    )


def build_pipeline() -> Pipeline:
    """Wire up all components and return a ready-to-use Pipeline."""
    _make_env()
    extractor = PdfExtractor()
    router = DefaultFileRouter(
        {
            "pdf": extractor,
            "txt": PlaintextExtractor(),
            "image": EasyOcrExtractor(_make_ocr_reader()),
        }
    )
    meta_extractor = ProgressNoteExtractor()
    llm = _make_llm()
    tasks = [
        _TASK_REGISTRY[name](llm)
        for name in settings.classifier_tasks
        if name in _TASK_REGISTRY
    ]
    classifier = LLMClassifier(tasks=tasks)
    return Pipeline(router=router, meta_extractor=meta_extractor, classifier=classifier)


def build_pair_pipeline() -> PairPipeline:
    """Wire up all components and return a ready-to-use PairPipeline."""
    _make_env()
    extractor = PdfExtractor()
    router = DefaultFileRouter(
        {
            "pdf": extractor,
            "txt": PlaintextExtractor(),
            "image": EasyOcrExtractor(_make_ocr_reader()),
        }
    )
    llm = _make_llm()
    return PairPipeline(
        router=router,
        dr_meta_extractor=ProgressNoteExtractor(),
        nurse_meta_extractor=NurseVisitExtractor(),
        pair_classifier=DrNurseMatchClassifier(llm=llm),
    )

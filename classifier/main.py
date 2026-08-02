import asyncio
import os
from functools import cache

from langchain_core.language_models import BaseChatModel

from classifier.classifiers.diagnosis_extraction import DiagnosisExtractionClassifier
from classifier.classifiers.doctor_visit_needed import DoctorVisitNeededClassifier
from classifier.classifiers.dr_nurse_match import DrNurseMatchClassifier
from classifier.classifiers.llm_classifier import LLMClassifier
from classifier.classifiers.treatment_request import TreatmentRequestClassifier
from classifier.dr_note_pipeline import DrNotePipeline
from classifier.extractors.easyocr_extractor import (
    EasyOcrExtractor,
    OcrReader,
    create_easyocr_reader,
)
from classifier.extractors.pdf import PdfExtractor
from classifier.extractors.plaintext import PlaintextExtractor
from classifier.metadata.nurse_visit import NurseVisitExtractor
from classifier.metadata.progress_note import ProgressNoteExtractor
from classifier.models import DiagnosisExtractionResult, TreatmentRequestResult
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


def _make_router() -> DefaultFileRouter:
    """Build the default file router mapping file types to content extractors."""
    return DefaultFileRouter(
        {
            "pdf": PdfExtractor(),
            "txt": PlaintextExtractor(),
            "image": EasyOcrExtractor(_make_ocr_reader()),
        }
    )


def build_diagnosis_extractor(llm: BaseChatModel | None = None) -> DiagnosisExtractionClassifier:
    """Wire up the diagnosis-extraction classifier; builds its own LLM if none given."""
    return DiagnosisExtractionClassifier(llm=llm or _make_llm())


def build_diagnosis_pipeline() -> DrNotePipeline[DiagnosisExtractionResult]:
    """Wire up a folder-capable diagnosis pipeline (dr notes only)."""
    _make_env()
    extractor = build_diagnosis_extractor()
    return DrNotePipeline(
        router=_make_router(),
        extract_fn=extractor.extract_diagnoses,
        error_fn=lambda path, _exc: DiagnosisExtractionResult(
            file_path=path, diagnoses=[]
        ),
    )


def build_treatment_request_extractor(
    llm: BaseChatModel | None = None,
) -> TreatmentRequestClassifier:
    """Wire up the treatment-request classifier; builds its own LLM if none given."""
    return TreatmentRequestClassifier(llm=llm or _make_llm())


def build_treatment_request_pipeline() -> DrNotePipeline[TreatmentRequestResult]:
    """Wire up a folder-capable treatment-request pipeline (dr notes only)."""
    _make_env()
    extractor = build_treatment_request_extractor()
    return DrNotePipeline(
        router=_make_router(),
        extract_fn=extractor.extract_treatment_request,
        error_fn=lambda path, exc: TreatmentRequestResult(
            file_path=path,
            treatment_requested=False,
            reasoning=f"Pipeline error: {exc}",
        ),
    )


def build_pipeline() -> Pipeline:
    """Wire up all components and return a ready-to-use Pipeline."""
    _make_env()
    meta_extractor = ProgressNoteExtractor()
    llm = _make_llm()
    tasks = [
        _TASK_REGISTRY[name](llm)
        for name in settings.classifier_tasks
        if name in _TASK_REGISTRY
    ]
    classifier = LLMClassifier(tasks=tasks)
    return Pipeline(
        router=_make_router(), meta_extractor=meta_extractor, classifier=classifier
    )


def build_pair_pipeline() -> PairPipeline:
    """Wire up all components and return a ready-to-use PairPipeline.

    The pair classifier gets a diagnosis extractor (pairs in
    ``settings.diagnosis_check_categories`` are checked for missing diagnoses) and a
    care extractor (pairs in ``settings.care_check_categories`` are checked for a
    treatment request or medical necessity). All three share one chat model, so the
    whole pipeline honours a single max_concurrent_llm_calls budget.
    """
    _make_env()
    llm = _make_llm()
    return PairPipeline(
        router=_make_router(),
        dr_meta_extractor=ProgressNoteExtractor(),
        nurse_meta_extractor=NurseVisitExtractor(),
        pair_classifier=DrNurseMatchClassifier(
            llm=llm,
            diagnosis_extractor=build_diagnosis_extractor(llm),
            care_extractor=build_treatment_request_extractor(llm),
        ),
    )

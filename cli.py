import argparse
import asyncio
import json
import logging
import shutil
import sys
from pathlib import Path

from classifier.ingest.sharepoint import SharePointFileUploader, SharePointFolderDownloader
from classifier.main import (
    build_diagnosis_pipeline,
    build_pair_pipeline,
    build_pipeline,
    build_treatment_request_pipeline,
)
from classifier.models import (
    DiagnosisExtractionResult,
    PairPipelineResult,
    PipelineResult,
    TreatmentRequestResult,
)
from classifier.naming import parse_note
from classifier.output.csv_writer import (
    relative_file_names,
    resolve_folder,
    write_diagnosis_xlsx,
    write_pair_xlsx,
    write_treatment_request_xlsx,
    write_xlsx,
)
from classifier.pair_pipeline import scan_pairs
from config.settings import settings

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def _result_to_dict(r: PipelineResult) -> dict:  # type: ignore[type-arg]
    if not r.success or r.result is None:
        return {"file": str(r.file_path), "success": False, "error": r.error}
    return {
        "file": str(r.file_path),
        "success": True,
        "category": r.result.category,
        "meta": r.result.metadata.meta.model_dump(),
        "hpi": r.result.metadata.hpi,
        "examination": r.result.metadata.examination.model_dump() if r.result.metadata.examination else None,
        "complaints": r.result.metadata.complaints,
        "medical_history": r.result.metadata.medical_history,
        "surgical_history": r.result.metadata.surgical_history,
        "hospitalization": r.result.metadata.hospitalization,
        "assessment": r.result.metadata.assessment,
        "ros": r.result.metadata.ros,
        "medications": r.result.metadata.medications.model_dump() if r.result.metadata.medications else None,
        "plan": r.result.metadata.plan,
        "procedure_codes": r.result.metadata.procedure_codes,
        "preventive_medicine": r.result.metadata.preventive_medicine,
        "task_results": r.result.task_results,
    }


def _pair_result_to_dict(
    r: PairPipelineResult,
    *,
    verbose: bool = False,
    local_root: Path | None = None,
    sp_web_url: str | None = None,
) -> dict:  # type: ignore[type-arg]
    dr_names = relative_file_names(r.dr_paths, r.dr_file_path, local_root)
    nurse_names = relative_file_names(r.nurse_paths, r.nurse_file_path, local_root)
    if not r.success or r.result is None:
        return {
            "dr_file": dr_names,
            "nurse_file": nurse_names,
            "success": False,
            "error": r.error,
        }
    res = r.result
    dr = res.dr_metadata
    nurse = res.nurse_fields
    out: dict = {  # type: ignore[type-arg]
        "folder": resolve_folder(r.dr_file_path, local_root, sp_web_url),
        "dr_file": dr_names,
        "nurse_file": nurse_names,
        "success": True,
        "overall": res.overall,
        "clinical_verdict": res.clinical_verdict,
        "clinical_reasoning": res.clinical_reasoning,
        "diagnosis_check": res.diagnosis_check or "",
        "diagnoses": [d.model_dump() for d in res.diagnoses],
        "care_check": res.care_check or "",
        "care_reasoning": res.care_reasoning,
    }
    if verbose:
        out["identity_match"] = res.identity_match.model_dump()
        out["dr_extracted"] = {
            "meta": dr.meta.model_dump(),
            "complaints": dr.complaints,
            "assessment": dr.assessment,
            "hpi": dr.hpi,
            "ros": dr.ros,
            "plan": dr.plan,
            "medications": dr.medications.model_dump() if dr.medications else None,
            "examination": dr.examination.model_dump() if dr.examination else None,
            "medical_history": dr.medical_history,
            "surgical_history": dr.surgical_history,
            "procedure_codes": dr.procedure_codes,
        }
        out["nurse_extracted"] = nurse.model_dump()
    return out


def _diagnosis_result_to_dict(r: DiagnosisExtractionResult) -> dict:  # type: ignore[type-arg]
    return {
        "file": str(r.file_path),
        "diagnoses": [d.model_dump() for d in r.diagnoses],
    }


async def _run_diagnosis(input_path: Path) -> tuple[int, Path | None]:
    """Extract diagnoses from a dr note file, or all dr notes in a folder."""
    pipeline = build_diagnosis_pipeline()
    if input_path.is_dir():
        results = await pipeline.run_folder(input_path)
    elif input_path.is_file():
        results = await pipeline.run([input_path])
    else:
        logger.error("Path does not exist: %s", input_path)
        return 1, None
    print(json.dumps([_diagnosis_result_to_dict(r) for r in results], indent=2, default=str))
    diagnosis_path = settings.output_path.parent / "diagnosis_results.xlsx"
    write_diagnosis_xlsx(results, diagnosis_path)
    return 0, diagnosis_path


def _treatment_result_to_dict(r: TreatmentRequestResult) -> dict:  # type: ignore[type-arg]
    return {
        "file": str(r.file_path),
        "treatment_requested": r.treatment_requested,
        "reasoning": r.reasoning,
    }


async def _run_treatment_request(input_path: Path) -> tuple[int, Path | None]:
    """Detect if anyone indicates treatment is needed, for a dr note file or all dr notes in a folder."""
    pipeline = build_treatment_request_pipeline()
    if input_path.is_dir():
        results = await pipeline.run_folder(input_path)
    elif input_path.is_file():
        results = await pipeline.run([input_path])
    else:
        logger.error("Path does not exist: %s", input_path)
        return 1, None
    print(json.dumps([_treatment_result_to_dict(r) for r in results], indent=2, default=str))
    out_path = settings.output_path.parent / "treatment_request_results.xlsx"
    write_treatment_request_xlsx(results, out_path)
    return 0, out_path


def resolve_results_folder(
    results_folder: str | None, sharepoint_folder: str | None
) -> str:
    """Pick the SharePoint folder the results xlsx is uploaded into.

    An explicit --results-folder wins; otherwise results land back in the
    --sharepoint-folder the notes were read from; local runs fall back to the
    configured results folder.
    """
    return results_folder or sharepoint_folder or settings.sharepoint_results_folder


def _is_pair_folder(path: Path) -> bool:
    """Return True if path contains any dr or nurse note (recursively, any category)."""
    return any(parse_note(f) is not None for f in path.rglob("*"))


async def _run(
    input_path: Path,
    *,
    verbose: bool = False,
    local_root: Path | None = None,
    sp_web_url: str | None = None,
) -> tuple[int, Path | None]:
    if input_path.is_dir() and _is_pair_folder(input_path):
        pipeline = build_pair_pipeline()
        pairs = scan_pairs(input_path)
        if not pairs:
            logger.error("No dr/nurse note pairs found in %s", input_path)
            return 1, None
        results = await pipeline.run_pairs(pairs)
        # Without SharePoint, the scanned folder is the root file paths are shown relative to.
        path_root = local_root or input_path
        output = [
            _pair_result_to_dict(r, verbose=verbose, local_root=path_root, sp_web_url=sp_web_url)
            for r in results
        ]
        print(json.dumps(output, indent=2, default=str))
        pair_path = settings.output_path.parent / "pair_results.xlsx"
        write_pair_xlsx(
            results, pair_path, verbose=verbose, local_root=path_root, sp_web_url=sp_web_url
        )
        return sum(1 for r in results if not r.success), pair_path

    pipeline = build_pipeline()
    if input_path.is_dir():
        single_results = await pipeline.run_folder(input_path)
    elif input_path.is_file():
        single_results = await pipeline.run([input_path])
    else:
        logger.error("Path does not exist: %s", input_path)
        return 1, None

    output = [_result_to_dict(r) for r in single_results]
    print(json.dumps(output, indent=2, default=str))
    write_xlsx(single_results, settings.output_path, task_names=settings.classifier_tasks)
    return sum(1 for r in single_results if not r.success), settings.output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="CobbleHill document classifier")
    parser.add_argument(
        "--input",
        default=settings.data_folder,
        required=False,
        type=str,
        help="Path to a local file or folder",
    )
    parser.add_argument(
        "--sharepoint-folder",
        default=None,
        type=str,
        help=(
            'Drive-relative SharePoint folder path, e.g. '
            '"Patient Encounters/Medical Notes/Non-Admits"'
        ),
    )
    parser.add_argument(
        "--download-dir",
        default=None,
        type=Path,
        help="Directory to download SharePoint files into (default: auto temp dir, deleted after run)",
    )
    parser.add_argument(
        "--upload-results",
        action="store_true",
        default=False,
        help=(
            "Upload the results xlsx to SharePoint after processing. Defaults to the "
            "--sharepoint-folder the notes came from, or "
            f"{settings.sharepoint_results_folder!r} for local runs"
        ),
    )
    parser.add_argument(
        "--results-folder",
        default=None,
        type=str,
        help=(
            "SharePoint folder to upload the results xlsx into "
            "(default: the --sharepoint-folder used for input)"
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Include identity_match, dr_fields, and nurse_fields columns in the output report",
    )
    parser.add_argument(
        "--diagnosis",
        action="store_true",
        default=False,
        help="Extract diagnoses from dr notes only (ignores nurse notes); prints JSON",
    )
    parser.add_argument(
        "--treatment-request",
        action="store_true",
        default=False,
        help="Detect if anyone indicates the patient should receive treatment (dr notes only); prints JSON",
    )
    args = parser.parse_args()

    _sp_kwargs = dict(
        client_id=settings.sharepoint_client_id,
        client_secret=settings.sharepoint_client_secret,
        tenant_id=settings.sharepoint_tenant_id,
        drive_id=settings.sharepoint_drive_id,
    )

    if args.sharepoint_folder:
        downloader = SharePointFolderDownloader(**_sp_kwargs)
        use_tmp = args.download_dir is None
        local_dir, sp_web_url = asyncio.run(downloader.download(args.sharepoint_folder, args.download_dir))
        try:
            if args.diagnosis:
                exit_code, written_csv = asyncio.run(_run_diagnosis(local_dir))
            elif args.treatment_request:
                exit_code, written_csv = asyncio.run(_run_treatment_request(local_dir))
            else:
                exit_code, written_csv = asyncio.run(
                    _run(local_dir, verbose=args.verbose, local_root=local_dir, sp_web_url=sp_web_url)
                )
        finally:
            if use_tmp:
                shutil.rmtree(local_dir, ignore_errors=True)
    elif args.diagnosis:
        exit_code, written_csv = asyncio.run(_run_diagnosis(Path(args.input)))
    elif args.treatment_request:
        exit_code, written_csv = asyncio.run(_run_treatment_request(Path(args.input)))
    else:
        exit_code, written_csv = asyncio.run(_run(Path(args.input), verbose=args.verbose))

    if args.upload_results and written_csv and written_csv.exists():
        uploader = SharePointFileUploader(**_sp_kwargs)
        folder = resolve_results_folder(args.results_folder, args.sharepoint_folder)
        asyncio.run(uploader.upload(written_csv, folder))

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
"""
# Results are uploaded back into the same SharePoint folder the notes came from
uv run python -m cli --sharepoint-folder "Patient Encounters/Medical Notes/Non-Admits" --upload-results

# Download to persistent dir
uv run python -m cli --sharepoint-folder "Patient Encounters/Medical Notes/Non-Admits" --download-dir ./downloads

# Upload somewhere else instead:
uv run python -m cli --sharepoint-folder "Patient Encounters/Medical Notes/Non-Admits" --upload-results --results-folder "Some/Other/Folder"

# Local input + upload: no --sharepoint-folder to inherit, so settings.sharepoint_results_folder is used
uv run python -m cli --input ./data --upload-results

# Local files (unchanged)
uv run python -m cli --input ./data
# Showing all fields
--verbose

# Diagnosis extraction (dr notes only, nurse notes ignored)
# Prints JSON and writes output/diagnosis_results.xlsx
uv run python -m cli --input ./data --diagnosis

# Single dr note file
uv run python -m cli --input ./data/1/dr_progress_note.pdf --diagnosis

# Treatment-request detection — true if ANYONE (patient, family, or provider)
# indicates the patient should be treated. dr notes only, nurse notes ignored.
# Prints JSON and writes output/treatment_request_results.xlsx
uv run python -m cli --input ./data --treatment-request

# Single dr note file
uv run python -m cli --input ./data/1/dr_progress_note.pdf --treatment-request

# From SharePoint, then upload the results xlsx back (works with --diagnosis or --treatment-request)
uv run python -m cli --sharepoint-folder "Patient Encounters/Medical Notes/Non-Admits" --diagnosis --upload-results
uv run python -m cli --sharepoint-folder "Patient Encounters/Medical Notes/Non-Admits" --treatment-request --upload-results

# Every LLM call goes to the configured ollama model, so ollama must be running.
# Point it elsewhere with COBBLEHILL_OLLAMA_URL / COBBLEHILL_OLLAMA_MODEL.

"""

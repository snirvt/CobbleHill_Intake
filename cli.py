import argparse
import asyncio
import json
import logging
import shutil
import sys
from pathlib import Path

from classifier.ingest.sharepoint import SharePointFileUploader, SharePointFolderDownloader
from classifier.main import build_pair_pipeline, build_pipeline
from classifier.models import PairPipelineResult, PipelineResult
from classifier.output.csv_writer import write_pair_xlsx, write_xlsx
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


def _pair_result_to_dict(r: PairPipelineResult, *, verbose: bool = False) -> dict:  # type: ignore[type-arg]
    if not r.success or r.result is None:
        return {
            "dr_file": str(r.dr_file_path),
            "nurse_file": str(r.nurse_file_path),
            "success": False,
            "error": r.error,
        }
    res = r.result
    dr = res.dr_metadata
    nurse = res.nurse_fields
    out: dict = {  # type: ignore[type-arg]
        "dr_file": str(r.dr_file_path),
        "nurse_file": str(r.nurse_file_path),
        "success": True,
        "overall": res.overall,
        "clinical_verdict": res.clinical_verdict,
        "clinical_reasoning": res.clinical_reasoning,
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


def _is_pair_folder(path: Path) -> bool:
    """Return True if path contains any dr_* or nurse_* supported files (recursively)."""
    exts = set(settings.supported_extensions.keys())
    for f in path.rglob("*"):
        if f.is_file() and f.suffix.lower() in exts:
            name = f.name.lower()
            if name.startswith("dr_") or name.startswith("nurse_"):
                return True
    return False


async def _run(input_path: Path, *, verbose: bool = False) -> tuple[int, Path | None]:
    if input_path.is_dir() and _is_pair_folder(input_path):
        pipeline = build_pair_pipeline()
        pairs = scan_pairs(input_path)
        if not pairs:
            logger.error("No dr_*/nurse_* pairs found in %s", input_path)
            return 1, None
        results = await pipeline.run_pairs(pairs)
        output = [_pair_result_to_dict(r, verbose=verbose) for r in results]
        print(json.dumps(output, indent=2, default=str))
        pair_path = settings.output_path.parent / "pair_results.xlsx"
        write_pair_xlsx(results, pair_path, verbose=verbose)
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
            f"Upload the results CSV to SharePoint after processing "
            f"(default folder: {settings.sharepoint_results_folder!r})"
        ),
    )
    parser.add_argument(
        "--results-folder",
        default=settings.sharepoint_results_folder,
        type=str,
        help="SharePoint folder to upload results CSV into",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Include identity_match, dr_fields, and nurse_fields columns in the output report",
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
        local_dir = asyncio.run(downloader.download(args.sharepoint_folder, args.download_dir))
        try:
            exit_code, written_csv = asyncio.run(_run(local_dir, verbose=args.verbose))
        finally:
            if use_tmp:
                shutil.rmtree(local_dir, ignore_errors=True)
    else:
        exit_code, written_csv = asyncio.run(_run(Path(args.input), verbose=args.verbose))

    if args.upload_results and written_csv and written_csv.exists():
        uploader = SharePointFileUploader(**_sp_kwargs)
        asyncio.run(uploader.upload(written_csv, args.results_folder))

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
"""
uv run python -m cli --sharepoint-folder "Patient Encounters/Medical Notes/Non-Admits" --upload-results

# Download to persistent dir
uv run python -m cli --sharepoint-folder "Patient Encounters/Medical Notes/Non-Admits" --download-dir ./downloads

# or override the results folder:
uv run python -m cli --input ./data --upload-results --results-folder "Some/Other/Folder"

# Local files (unchanged)
uv run python -m cli --input ./data
# Showing all fields
--verbose

"""

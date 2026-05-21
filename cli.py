import argparse
import asyncio
import json
import logging
import shutil
import sys
from pathlib import Path

from classifier.ingest.sharepoint import SharePointFolderDownloader, is_sharepoint_url
from classifier.main import build_pair_pipeline, build_pipeline
from classifier.models import PairPipelineResult, PipelineResult
from classifier.output.csv_writer import write_csv, write_pair_csv
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


def _pair_result_to_dict(r: PairPipelineResult) -> dict:  # type: ignore[type-arg]
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
    return {
        "dr_file": str(r.dr_file_path),
        "nurse_file": str(r.nurse_file_path),
        "success": True,
        "overall": res.overall,
        "clinical_verdict": res.clinical_verdict,
        "clinical_reasoning": res.clinical_reasoning,
        "identity_match": res.identity_match.model_dump(),
        "dr_extracted": {
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
        },
        "nurse_extracted": nurse.model_dump(),
    }


def _is_pair_folder(path: Path) -> bool:
    """Return True if path contains any dr_* or nurse_* supported files (recursively)."""
    exts = set(settings.supported_extensions.keys())
    for f in path.rglob("*"):
        if f.is_file() and f.suffix.lower() in exts:
            name = f.name.lower()
            if name.startswith("dr_") or name.startswith("nurse_"):
                return True
    return False


async def _run(input_path: Path, pair_csv: Path) -> int:
    if input_path.is_dir() and _is_pair_folder(input_path):
        pipeline = build_pair_pipeline()
        pairs = scan_pairs(input_path)
        if not pairs:
            logger.error("No dr_*/nurse_* pairs found in %s", input_path)
            return 1
        results = await pipeline.run_pairs(pairs)
        output = [_pair_result_to_dict(r) for r in results]
        print(json.dumps(output, indent=2, default=str))
        write_pair_csv(results, pair_csv)
        return sum(1 for r in results if not r.success)

    pipeline = build_pipeline()
    if input_path.is_dir():
        single_results = await pipeline.run_folder(input_path)
    elif input_path.is_file():
        single_results = await pipeline.run([input_path])
    else:
        logger.error("Path does not exist: %s", input_path)
        return 1

    output = [_result_to_dict(r) for r in single_results]
    print(json.dumps(output, indent=2, default=str))
    write_csv(single_results, settings.output_csv, task_names=settings.classifier_tasks)
    return sum(1 for r in single_results if not r.success)


def main() -> None:
    parser = argparse.ArgumentParser(description="CobbleHill document classifier")
    parser.add_argument(
        "--input",
        default=settings.data_folder,
        required=False,
        type=str,
        help="Path to a single PDF/folder, or a SharePoint folder URL",
    )
    parser.add_argument(
        "--pair-csv",
        type=Path,
        default=settings.output_csv.parent / "pair_results.csv",
        help="Output CSV path for pair comparison results",
    )
    args = parser.parse_args()

    if is_sharepoint_url(args.input):
        client_id = settings.sharepoint_client_id
        if not client_id:
            logger.error(
                "SharePoint URL given but COBBLEHILL_SHAREPOINT_CLIENT_ID is not set"
            )
            sys.exit(1)
        downloader = SharePointFolderDownloader(client_id)
        tmp_dir: Path | None = None
        try:
            tmp_dir = downloader.download_to_temp(args.input)
            sys.exit(asyncio.run(_run(tmp_dir, args.pair_csv)))
        finally:
            if tmp_dir is not None:
                shutil.rmtree(tmp_dir, ignore_errors=True)
    else:
        sys.exit(asyncio.run(_run(Path(args.input), args.pair_csv)))


if __name__ == "__main__":
    main()

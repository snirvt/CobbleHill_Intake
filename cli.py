import argparse
import asyncio
import json
import logging
import shutil
import sys
from pathlib import Path, PurePosixPath

from classifier.ingest.sharepoint import SharePointFileUploader, SharePointFolderDownloader
from classifier.main import (
    build_diagnosis_pipeline,
    build_dr_validity_pipeline,
    build_main_diagnosis_pipeline,
    build_pair_pipeline,
    build_pipeline,
    build_treatment_request_pipeline,
)
from classifier.models import (
    DiagnosisExtractionResult,
    MainDiagnosisResult,
    PipelineResult,
    TreatmentRequestResult,
)
from classifier.naming import parse_note
from classifier.output.csv_writer import (
    write_diagnosis_xlsx,
    write_dr_validity_xlsx,
    write_main_diagnosis_xlsx,
    write_pair_xlsx,
    write_treatment_request_xlsx,
    write_xlsx,
)
from classifier.output.dr_validity_report import dr_validity_patient_dict
from classifier.output.pair_report import pair_dict
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


def _diagnosis_result_to_dict(r: DiagnosisExtractionResult) -> dict:  # type: ignore[type-arg]
    return {
        "file": str(r.file_path),
        "diagnoses": [d.model_dump() for d in r.diagnoses],
        "reasoning": r.reasoning,
    }


async def _run_diagnosis(input_path: Path, *, out_dir: Path) -> tuple[int, Path | None]:
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
    diagnosis_path = out_dir / "diagnosis_results.xlsx"
    write_diagnosis_xlsx(results, diagnosis_path)
    return 0, diagnosis_path


def _treatment_result_to_dict(r: TreatmentRequestResult) -> dict:  # type: ignore[type-arg]
    return {
        "file": str(r.file_path),
        "treatment_requested": r.treatment_requested,
        "reasoning": r.reasoning,
    }


async def _run_treatment_request(
    input_path: Path, *, out_dir: Path
) -> tuple[int, Path | None]:
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
    out_path = out_dir / "treatment_request_results.xlsx"
    write_treatment_request_xlsx(results, out_path)
    return 0, out_path


def _main_diagnosis_result_to_dict(r: MainDiagnosisResult) -> dict:  # type: ignore[type-arg]
    return {
        "file": str(r.file_path),
        "main_diagnosis": r.main_diagnosis,
        "diagnosis_category": r.diagnosis_category,
        "icd_code": r.icd_code,
        "reasoning": r.reasoning,
    }


async def _run_main_diagnosis(
    input_path: Path, *, out_dir: Path
) -> tuple[int, Path | None]:
    """Extract a short main-diagnosis label, for a dr note file or all dr notes in a folder."""
    pipeline = build_main_diagnosis_pipeline()
    if input_path.is_dir():
        results = await pipeline.run_folder(input_path)
    elif input_path.is_file():
        results = await pipeline.run([input_path])
    else:
        logger.error("Path does not exist: %s", input_path)
        return 1, None
    print(json.dumps([_main_diagnosis_result_to_dict(r) for r in results], indent=2, default=str))
    out_path = out_dir / "main_diagnosis_results.xlsx"
    write_main_diagnosis_xlsx(results, out_path)
    return 0, out_path


async def _run_dr_validity(
    input_path: Path,
    *,
    out_dir: Path,
    local_root: Path | None = None,
    sp_web_url: str | None = None,
    root_name: str | None = None,
) -> tuple[int, Path | None]:
    """Check each patient folder's dr notes (combined) against the intake rules.

    input_path may be a folder of patient folders or a single patient folder. root_name
    names the patient whose notes sit directly in input_path (e.g. the SharePoint folder
    name, since a download lands in a temp dir).
    """
    if not input_path.is_dir():
        logger.error("--dr-validity needs a folder of patient folders: %s", input_path)
        return 1, None
    results = await build_dr_validity_pipeline().run_folder(input_path)
    path_root = local_root or input_path
    output = [dr_validity_patient_dict(r, path_root, root_name) for r in results]
    print(json.dumps(output, indent=2, default=str))
    out_path = out_dir / "dr_validity_results.xlsx"
    write_dr_validity_xlsx(
        results, out_path, local_root=path_root, sp_web_url=sp_web_url, root_name=root_name
    )
    return sum(1 for r in results if not r.success), out_path


def resolve_results_folder(
    upload_folder: str | None, sharepoint_folder: str | None
) -> str:
    """Pick the SharePoint folder the results xlsx is uploaded into.

    An explicit --upload-results folder wins; otherwise results land in a subfolder of the
    --sharepoint-folder the notes were read from; local runs, which have no input
    folder to inherit, fall back to the configured results folder.
    """
    if upload_folder:
        return upload_folder
    if sharepoint_folder:
        return f"{sharepoint_folder.rstrip('/')}/{settings.sharepoint_results_subfolder}"
    return settings.sharepoint_results_folder


def _is_pair_folder(path: Path) -> bool:
    """Return True if path contains any dr or nurse note (recursively, any category)."""
    return any(parse_note(f) is not None for f in path.rglob("*"))


async def _run_dr_nurse_match(
    input_path: Path,
    *,
    out_dir: Path,
    verbose: bool = False,
    local_root: Path | None = None,
    sp_web_url: str | None = None,
) -> tuple[int, Path | None]:
    """Compare every same-category dr + nurse pair in a folder; print JSON, write xlsx.

    Folders without a pair are logged as warnings, not processed. Returns (number of
    failed pairs, xlsx path); (1, None) when input_path is not a folder.
    """
    if not input_path.is_dir():
        logger.error("Not a folder: %s", input_path)
        return 1, None
    pipeline = build_pair_pipeline()
    results = await pipeline.run_folder(input_path)
    for error in pipeline.folder_errors(input_path):
        logger.warning("Skipped folder %s: %s", error.folder, error.message)
    # Without SharePoint, the scanned folder is the root file paths are shown relative to.
    path_root = local_root or input_path
    output = [
        pair_dict(r, verbose=verbose, local_root=path_root, sp_web_url=sp_web_url)
        for r in results
    ]
    print(json.dumps(output, indent=2, default=str))
    pair_path = out_dir / "pair_results.xlsx"
    write_pair_xlsx(
        results, pair_path, verbose=verbose, local_root=path_root, sp_web_url=sp_web_url
    )
    return sum(1 for r in results if not r.success), pair_path


async def _run(
    input_path: Path,
    *,
    out_dir: Path,
    verbose: bool = False,
    local_root: Path | None = None,
    sp_web_url: str | None = None,
) -> tuple[int, Path | None]:
    if input_path.is_dir() and _is_pair_folder(input_path):
        if not scan_pairs(input_path):
            logger.error("No dr/nurse note pairs found in %s", input_path)
            return 1, None
        return await _run_dr_nurse_match(
            input_path,
            out_dir=out_dir,
            verbose=verbose,
            local_root=local_root,
            sp_web_url=sp_web_url,
        )

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
    out_path = out_dir / settings.output_path.name
    write_xlsx(single_results, out_path, task_names=settings.classifier_tasks)
    return sum(1 for r in single_results if not r.success), out_path


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
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
        nargs="?",
        const="",
        default=None,
        metavar="SP_FOLDER",
        help=(
            "Upload the results xlsx to SharePoint after processing, into SP_FOLDER if given. "
            f"Default: the {settings.sharepoint_results_subfolder!r} subfolder of "
            f"--sharepoint-folder, or {settings.sharepoint_results_folder!r} for local runs"
        ),
    )
    parser.add_argument(
        "--results-folder",
        default=None,
        type=Path,
        metavar="LOCAL_DIR",
        help=(
            "Local directory to save the results xlsx into "
            f"(default: {settings.output_path.parent})"
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
    parser.add_argument(
        "--main-diagnosis",
        action="store_true",
        default=False,
        help="Extract a short (1-4 word) main diagnosis per dr note, to group cases; prints JSON",
    )
    parser.add_argument(
        "--dr-validity",
        action="store_true",
        default=False,
        help=(
            "Check each patient folder's dr notes: under 18 needs diagnosis + medical "
            "necessity, adults need a diagnosis; also labels the main diagnosis"
        ),
    )
    parser.add_argument(
        "--dr-nurse-match",
        action="store_true",
        default=False,
        help=(
            "Compare each same-category dr + nurse note pair (clinical match, diagnosis/care "
            "checks, main diagnosis); folders without a pair are reported"
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out_dir: Path = args.results_folder or settings.output_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

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
                exit_code, written_csv = asyncio.run(_run_diagnosis(local_dir, out_dir=out_dir))
            elif args.treatment_request:
                exit_code, written_csv = asyncio.run(
                    _run_treatment_request(local_dir, out_dir=out_dir)
                )
            elif args.main_diagnosis:
                exit_code, written_csv = asyncio.run(_run_main_diagnosis(local_dir, out_dir=out_dir))
            elif args.dr_validity:
                exit_code, written_csv = asyncio.run(
                    _run_dr_validity(
                        local_dir,
                        out_dir=out_dir,
                        local_root=local_dir,
                        sp_web_url=sp_web_url,
                        root_name=PurePosixPath(args.sharepoint_folder.rstrip("/")).name,
                    )
                )
            elif args.dr_nurse_match:
                exit_code, written_csv = asyncio.run(
                    _run_dr_nurse_match(
                        local_dir,
                        out_dir=out_dir,
                        verbose=args.verbose,
                        local_root=local_dir,
                        sp_web_url=sp_web_url,
                    )
                )
            else:
                exit_code, written_csv = asyncio.run(
                    _run(
                        local_dir,
                        out_dir=out_dir,
                        verbose=args.verbose,
                        local_root=local_dir,
                        sp_web_url=sp_web_url,
                    )
                )
        finally:
            if use_tmp:
                shutil.rmtree(local_dir, ignore_errors=True)
    elif args.diagnosis:
        exit_code, written_csv = asyncio.run(
            _run_diagnosis(Path(args.input), out_dir=out_dir)
        )
    elif args.treatment_request:
        exit_code, written_csv = asyncio.run(
            _run_treatment_request(Path(args.input), out_dir=out_dir)
        )
    elif args.main_diagnosis:
        exit_code, written_csv = asyncio.run(
            _run_main_diagnosis(Path(args.input), out_dir=out_dir)
        )
    elif args.dr_validity:
        exit_code, written_csv = asyncio.run(
            _run_dr_validity(Path(args.input), out_dir=out_dir)
        )
    elif args.dr_nurse_match:
        exit_code, written_csv = asyncio.run(
            _run_dr_nurse_match(Path(args.input), out_dir=out_dir, verbose=args.verbose)
        )
    else:
        exit_code, written_csv = asyncio.run(
            _run(Path(args.input), out_dir=out_dir, verbose=args.verbose)
        )

    if args.upload_results is not None and written_csv and written_csv.exists():
        uploader = SharePointFileUploader(**_sp_kwargs)
        folder = resolve_results_folder(args.upload_results or None, args.sharepoint_folder)
        asyncio.run(uploader.upload(written_csv, folder))

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
"""
# Results are uploaded into a "results" subfolder of the SharePoint folder the notes came from,
# e.g. "Patient Encounters/Medical Notes/Non-Admits/results"
uv run python -m cli --sharepoint-folder "Patient Encounters/Medical Notes/Non-Admits" --upload-results

# Download to persistent dir
uv run python -m cli --sharepoint-folder "Patient Encounters/Medical Notes/Non-Admits" --download-dir ./downloads

# Upload somewhere else instead (SharePoint path given to --upload-results):
uv run python -m cli --sharepoint-folder "Patient Encounters/Medical Notes/Non-Admits" --upload-results "Some/Other/Folder"

# Save the results xlsx to a local directory other than output/ (independent of the upload path):
uv run python -m cli --input ./data --results-folder ./my_output
uv run python -m cli --input ./data --results-folder ./my_output --upload-results "Some/Other/Folder"

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

# Main diagnosis — one short (1-2 word) label per dr note, for grouping cases.
# dr notes only. Prints JSON and writes output/main_diagnosis_results.xlsx
# (sheets: MainDiagnosis per file, Groups with counts). The default pair run
# also adds a main_diagnosis column to pair_results.xlsx.
uv run python -m cli --input ./data --main-diagnosis

# Dr-note validity — one row per patient folder, all of its dr_ notes combined.
# Under 18: needs a diagnosis AND medical necessity; 18+: needs a diagnosis.
# Writes output/dr_validity_results.xlsx (All / Valid / Needs Review / Groups)
uv run python -m cli --input ./data --dr-validity
uv run python -m cli --sharepoint-folder "Patient Encounters/Physician Notes/Pending" \
    --dr-validity --upload-results
# Same, saved locally to ./out and uploaded to a chosen SharePoint folder
uv run python -m cli --sharepoint-folder "Patient Encounters/Physician Notes/Pending" \
    --dr-validity --results-folder ./out --upload-results "Patient Encounters/Physician Notes/Reviewed"

# Same check over HTTP (FastAPI, localhost only, no auth). Blocks until done, returns JSON:
uv run cobblehill-api
# Reachable from the local network — needs an API key (clients send X-API-Key):
# Key = API_TOKEN in _SECRETS/_secrets.py
uv run cobblehill-api --host 0.0.0.0 --port 8010
curl -H "X-API-Key: <key>" -X POST localhost:8010/dr-validity -H 'Content-Type: application/json' \
    -d '{"sharepoint_folder": "Patient Encounters/Physician Notes/Pending"}'

# From SharePoint, then upload the results xlsx back
# (works with --diagnosis, --treatment-request, --main-diagnosis or --dr-validity)
uv run python -m cli --sharepoint-folder "Patient Encounters/Medical Notes/Non-Admits" --diagnosis --upload-results
uv run python -m cli --sharepoint-folder "Patient Encounters/Medical Notes/Non-Admits" --treatment-request --upload-results
uv run python -m cli --sharepoint-folder "Patient Encounters/Medical Notes/Non-Admits" --main-diagnosis --upload-results

# Every LLM call goes to the configured ollama model, so ollama must be running.
# Point it elsewhere with COBBLEHILL_OLLAMA_URL / COBBLEHILL_OLLAMA_MODEL.









uv run cobblehill-api --host 0.0.0.0        # network; port 8010 default
uv run cobblehill-api                        # localhost only
uv run cobblehill-api --host 0.0.0.0 --port 9000

Call endpoints. Every route except /health needs the key: by default that's API_TOKEN in _SECRETS/_secrets.py, or set it with the COBBLEHILL_API_KEY env var.

KEY=<API_TOKEN value>
HOST=localhost:8010   # or <spark-ip>:8010 from another machine

# liveness
curl http://$HOST/health

# dr-validity (existing)
curl -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
     -d '{"sharepoint_folder": "Patient Encounters/Physician Notes/Pending"}' \
     http://$HOST/dr-validity

# dr-nurse match (new)
curl -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
     -d '{"sharepoint_folder": "Patient Encounters/Physician Notes/Pending"}' \
     http://$HOST/dr-nurse-match

"""

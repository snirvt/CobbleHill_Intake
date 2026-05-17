import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from classifier.main import build_pipeline
from classifier.models import PipelineResult
from classifier.output.csv_writer import write_csv
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


async def _run(input_path: Path) -> int:
    pipeline = build_pipeline()

    if input_path.is_dir():
        results = await pipeline.run_folder(input_path)
    elif input_path.is_file():
        results = await pipeline.run([input_path])
    else:
        logger.error("Path does not exist: %s", input_path)
        return 1

    output = [_result_to_dict(r) for r in results]
    print(json.dumps(output, indent=2, default=str))

    write_csv(results, settings.output_csv, task_names=settings.classifier_tasks)

    failed = sum(1 for r in results if not r.success)
    return failed


def main() -> None:
    parser = argparse.ArgumentParser(description="CobbleHill document classifier")
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to a single PDF file or a folder containing PDF files",
    )
    args = parser.parse_args()
    sys.exit(asyncio.run(_run(args.input)))


if __name__ == "__main__":
    main()

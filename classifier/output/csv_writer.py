import csv
import logging
from pathlib import Path

from classifier.models import PipelineResult

logger = logging.getLogger(__name__)

# Fixed patient metadata columns in output order
_META_COLUMNS = [
    "patient_name",
    "dob",
    "age",
    "sex",
    "account_number",
    "dos",
    "phone",
    "address",
]


def write_csv(results: list[PipelineResult], output_path: Path, task_names: list[str]) -> None:
    """Write pipeline results to a CSV file.

    Columns: file_path, success, category, <one per task>, <patient meta fields>, errors
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["file_path", "success", "category"] + task_names + _META_COLUMNS + ["errors"]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(_result_to_row(result, task_names))

    logger.info("CSV written to %s (%d rows)", output_path, len(results))


def _result_to_row(result: PipelineResult, task_names: list[str]) -> dict[str, object]:
    base: dict[str, object] = {
        "file_path": str(result.file_path),
        "success": result.success,
        "errors": result.error or "",
    }

    if not result.success or result.result is None:
        base["category"] = ""
        for task in task_names:
            base[task] = ""
        for col in _META_COLUMNS:
            base[col] = ""
        return base

    meta = result.result.metadata.meta
    base["category"] = result.result.category
    for task in task_names:
        base[task] = result.result.task_results.get(task, "")
    for col in _META_COLUMNS:
        base[col] = getattr(meta, col) or ""
    return base

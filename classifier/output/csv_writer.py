import csv
import json
import logging
from pathlib import Path

from classifier.models import PairPipelineResult, PipelineResult

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

_PAIR_IDENTITY_COLUMNS = [
    "identity_patient_name",
    "identity_dob",
    "identity_dos",
    "identity_sex",
    "identity_account_number",
]

_PAIR_COLUMNS = (
    ["dr_file_path", "nurse_file_path", "success", "overall", "clinical_verdict", "clinical_reasoning"]
    + _PAIR_IDENTITY_COLUMNS
    + ["dr_fields", "nurse_fields", "errors"]
)


def write_pair_csv(results: list[PairPipelineResult], output_path: Path) -> None:
    """Write pair pipeline results to a CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_PAIR_COLUMNS)
        writer.writeheader()
        for result in results:
            writer.writerow(_pair_result_to_row(result))
    logger.info("Pair CSV written to %s (%d rows)", output_path, len(results))


def _pair_result_to_row(result: PairPipelineResult) -> dict[str, object]:
    base: dict[str, object] = {
        "dr_file_path": str(result.dr_file_path),
        "nurse_file_path": str(result.nurse_file_path),
        "success": result.success,
        "errors": result.error or "",
    }
    if not result.success or result.result is None:
        base.update({
            "overall": "",
            "clinical_verdict": "",
            "clinical_reasoning": "",
            **{col: "" for col in _PAIR_IDENTITY_COLUMNS},
            "dr_fields": "",
            "nurse_fields": "",
        })
        return base

    r = result.result
    base.update({
        "overall": r.overall,
        "clinical_verdict": r.clinical_verdict,
        "clinical_reasoning": r.clinical_reasoning,
        "identity_patient_name": r.identity_match.patient_name,
        "identity_dob": r.identity_match.dob,
        "identity_dos": r.identity_match.dos,
        "identity_sex": r.identity_match.sex,
        "identity_account_number": r.identity_match.account_number,
        "dr_fields": json.dumps(r.dr_metadata.model_dump(exclude={"file_path", "raw_text"}), default=str),
        "nurse_fields": json.dumps(r.nurse_fields.model_dump(), default=str),
    })
    return base


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

from pathlib import Path

import pytest

from classifier.classifiers.step_errors import raise_on_step_errors
from classifier.models import ClassificationStepError, ErrorCode, ErrorStage


def test_no_error_when_every_step_succeeded() -> None:
    raise_on_step_errors(Path("p"), {ErrorStage.AGE: None, ErrorStage.CARE: ""})


def test_raises_with_one_run_error_per_failed_step() -> None:
    with pytest.raises(ClassificationStepError) as exc_info:
        raise_on_step_errors(
            Path("patient"),
            {
                ErrorStage.AGE: None,
                ErrorStage.DIAGNOSIS: "Parse error: bad json",
                ErrorStage.CLINICAL_MATCH: "LLM error: timeout",
            },
        )
    errors = exc_info.value.errors
    assert [(e.stage, e.code, e.message, e.folder) for e in errors] == [
        (ErrorStage.DIAGNOSIS, ErrorCode.PARSE_FAILED, "Parse error: bad json", Path("patient")),
        (ErrorStage.CLINICAL_MATCH, ErrorCode.LLM_FAILED, "LLM error: timeout", Path("patient")),
    ]
    assert str(exc_info.value) == (
        "diagnosis: Parse error: bad json; clinical_match: LLM error: timeout"
    )

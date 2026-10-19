"""Turn failed classifier steps into a ClassificationStepError, shared by every classifier."""

from pathlib import Path

from classifier.models import ClassificationStepError, ErrorCode, ErrorStage, RunError

# Extractors prefix their error messages with this on output-parse failures.
PARSE_ERROR_PREFIX = "Parse error"


def raise_on_step_errors(folder: Path, step_errors: dict[ErrorStage, str | None]) -> None:
    """Raise ClassificationStepError listing every step whose error message is set.

    A message starting with ``PARSE_ERROR_PREFIX`` is a parse failure; any other is an
    LLM failure. Does nothing when no step failed.
    """
    errors = [
        RunError(
            stage=stage,
            code=ErrorCode.PARSE_FAILED
            if message.startswith(PARSE_ERROR_PREFIX)
            else ErrorCode.LLM_FAILED,
            message=message,
            folder=folder,
        )
        for stage, message in step_errors.items()
        if message
    ]
    if errors:
        raise ClassificationStepError(errors)

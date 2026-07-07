"""Evaluate DrNurseMatchClassifier against labeled file pairs.

Usage:
    python tests/eval/run_dr_nurse_match.py

Add labeled examples in tests/eval/datasets/dr_nurse_match.py before running.
"""

import asyncio
import logging
import sys
from pathlib import Path

from classifier.main import build_pair_pipeline
from tests.eval.datasets.dr_nurse_match import EXAMPLES
from tests.eval.harness import EvalHarness, LabeledExample

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")


async def main() -> None:
    if not EXAMPLES:
        print(
            "No examples found.\n"
            "Add labeled PairExample entries to tests/eval/datasets/dr_nurse_match.py"
        )
        sys.exit(0)

    pipeline = build_pair_pipeline()

    async def predict(pair: tuple[list[Path], list[Path]]) -> str:
        dr_paths, nurse_paths = pair
        results = await pipeline.run_pairs([(dr_paths, nurse_paths)])
        r = results[0]
        if not r.success or r.result is None:
            raise RuntimeError(r.error or "pipeline returned no result")
        return r.result.overall

    examples: list[LabeledExample[tuple[list[Path], list[Path]]]] = [
        LabeledExample(
            input=(ex.dr_paths, ex.nurse_paths),
            expected=ex.expected,
            description=ex.description
            or f"{'; '.join(p.name for p in ex.dr_paths)} + {'; '.join(p.name for p in ex.nurse_paths)}",
        )
        for ex in EXAMPLES
    ]

    harness: EvalHarness[tuple[list[Path], list[Path]]] = EvalHarness(
        classifier_name="dr_nurse_match",
        examples=examples,
        predict_fn=predict,
    )
    report = await harness.run()
    report.print_report()


if __name__ == "__main__":
    asyncio.run(main())

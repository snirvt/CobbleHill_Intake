import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class LabeledExample[T]:
    """One labeled input/output pair for evaluation."""

    input: T
    expected: str
    description: str = ""


@dataclass
class ExampleResult[T]:
    """Outcome of running predict_fn on one LabeledExample."""

    example: LabeledExample[T]
    predicted: str | None
    error: str | None = None

    @property
    def correct(self) -> bool:
        return self.predicted == self.example.expected


@dataclass
class ClassMetrics:
    label: str
    precision: float
    recall: float
    f1: float
    support: int


@dataclass
class EvalReport[T]:
    classifier_name: str
    results: list[ExampleResult[T]]
    per_class: dict[str, ClassMetrics]
    macro_precision: float
    macro_recall: float
    macro_f1: float
    accuracy: float

    def print_report(self) -> None:
        total = len(self.results)
        errors = sum(1 for r in self.results if r.error is not None)
        correct = sum(1 for r in self.results if r.correct)

        print(f"\n{'=' * 60}")
        print(f"Evaluation Report: {self.classifier_name}")
        print(f"{'=' * 60}")
        print(f"Total: {total}  |  Correct: {correct}  |  Errors: {errors}")
        print(f"Accuracy: {self.accuracy:.3f}")
        print()

        print(f"{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
        print("-" * 58)
        for m in self.per_class.values():
            print(
                f"{m.label:<15} {m.precision:>10.3f} {m.recall:>10.3f}"
                f" {m.f1:>10.3f} {m.support:>10}"
            )
        print("-" * 58)
        print(
            f"{'macro avg':<15} {self.macro_precision:>10.3f}"
            f" {self.macro_recall:>10.3f} {self.macro_f1:>10.3f}"
        )
        print()

        print("Individual results:")
        for r in self.results:
            tag = "[ERROR]" if r.error else ("[PASS]" if r.correct else "[FAIL]")
            desc = r.example.description or str(r.example.input)
            print(f"  {tag} {desc}")
            if r.error:
                print(f"         Error: {r.error}")
            else:
                print(f"         Expected: {r.example.expected!r:<12} Got: {r.predicted!r}")


class EvalHarness[T]:
    """Generic evaluation harness for any classifier that maps T → string label.

    To add a new classifier: create a dataset file + runner script. No changes here.
    """

    def __init__(
        self,
        classifier_name: str,
        examples: list[LabeledExample[T]],
        predict_fn: Callable[[T], Awaitable[str]],
    ) -> None:
        self._name = classifier_name
        self._examples = examples
        self._predict_fn = predict_fn

    async def run(self) -> EvalReport[T]:
        results: list[ExampleResult[T]] = list(
            await asyncio.gather(*[self._run_one(ex) for ex in self._examples])
        )
        return self._compute_report(results)

    async def _run_one(self, example: LabeledExample[T]) -> ExampleResult[T]:
        try:
            predicted = await self._predict_fn(example.input)
            return ExampleResult(example=example, predicted=predicted)
        except Exception as exc:
            label = example.description or str(example.input)
            logger.error("Prediction failed for %s: %s", label, exc)
            return ExampleResult(example=example, predicted=None, error=str(exc))

    def _compute_report(self, results: list[ExampleResult[T]]) -> EvalReport[T]:
        classes = sorted({ex.expected for ex in self._examples})
        per_class: dict[str, ClassMetrics] = {}

        for cls in classes:
            tp = sum(1 for r in results if r.example.expected == cls and r.predicted == cls)
            fp = sum(1 for r in results if r.example.expected != cls and r.predicted == cls)
            fn = sum(1 for r in results if r.example.expected == cls and r.predicted != cls)
            support = sum(1 for r in results if r.example.expected == cls)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

            per_class[cls] = ClassMetrics(
                label=cls, precision=precision, recall=recall, f1=f1, support=support
            )

        n_cls = len(per_class)
        macro_p = sum(m.precision for m in per_class.values()) / n_cls if n_cls else 0.0
        macro_r = sum(m.recall for m in per_class.values()) / n_cls if n_cls else 0.0
        macro_f1 = sum(m.f1 for m in per_class.values()) / n_cls if n_cls else 0.0

        total = len(results)
        accuracy = sum(1 for r in results if r.correct) / total if total else 0.0

        return EvalReport(
            classifier_name=self._name,
            results=results,
            per_class=per_class,
            macro_precision=macro_p,
            macro_recall=macro_r,
            macro_f1=macro_f1,
            accuracy=accuracy,
        )

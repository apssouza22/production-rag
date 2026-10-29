"""Small, dependency-free example of the shared evaluation infrastructure."""

import asyncio
import json
from typing import Any

from evals.core import (
    Dataset,
    EvalRunner,
    EvalSample,
    EvaluationReport,
    Evaluator,
    MetricDefinition,
    MetricResult,
    SampleField,
    build_report,
    get_dataset_source_name,
    write_report,
)


class ExampleDataset:
    """Provide one inline sample so the example needs no external services."""

    source_name = "inline_example"
    total_items = 1

    async def load(self) -> list[EvalSample]:
        return [
            EvalSample(
                input="What is the capital of France?",
                output="Paris",
                reference="Paris",
                source_id="example-1",
            )
        ]


async def score_exact_match(sample: EvalSample) -> MetricResult:
    """Score whether the generated output exactly matches the reference."""
    is_match = (sample.output or "").strip().casefold() == (sample.reference or "").strip().casefold()
    return MetricResult(
        name="exact_match",
        value=float(is_match),
        reasoning="Output matches the reference." if is_match else "Output does not match the reference.",
    )


class ExampleEvaluationRunner(EvalRunner):
    """Wire an example dataset and metric into the shared evaluator."""

    def __init__(self, dataset: Dataset | None = None) -> None:
        self.dataset = dataset or ExampleDataset()
        self.metric = MetricDefinition(
            name="exact_match",
            description="Whether output and reference are equal after normalization.",
            scorer=score_exact_match,
            required_fields=frozenset({SampleField.OUTPUT, SampleField.REFERENCE}),
        )
        self.evaluator = Evaluator([self.metric], progress_description="Running example evaluation")
        self.report: EvaluationReport | None = None

    async def run_eval(self, generate_report_file: bool = True) -> dict[str, Any]:
        response = await self.evaluator.evaluate(self.dataset)
        self.report = build_report(
            response=response,
            run_type="example_evaluation",
            evaluation_model="deterministic",
            dataset_source=get_dataset_source_name(self.dataset),
        )
        if generate_report_file:
            write_report(self.report)
        return self.report.model_dump(mode="json")


async def run() -> None:
    """Run the example and print its unified report."""
    report = await ExampleEvaluationRunner().run_eval(generate_report_file=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    asyncio.run(run())

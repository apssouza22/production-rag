# Add shared eval core (normalized samples + metric gating) and an offline example runner

**Task ID:** `01a17400-fb50-7004-84fa-df2ff1a42250`

## Summary

At base commit `f030fe10`, this repository already runs evaluations in two separate places: `evals/custom_evals/` (Langfuse trace scoring with OpenAI judges) and `evals/rag_evals/` (Ragas over rerank-search samples). Each path defines its own sample types and loops metrics in the runner—`RagEvaluator._evaluate_sample` only special-cases missing **reference** before calling `await metric.scorer(sample)`. There is no shared `evals/core/` package and no `evals/example_eval/`.

Add **`evals/core/`** with provider-neutral contracts that future eval paths can import, plus **`evals/example_eval/`** as a minimal offline runner. Do **not** rewrite `custom_evals` or `rag_evals` in this task; the new code should mirror field names already used in `rag_evals` (`user_input` / `response` / `retrieved_contexts`) via aliases on a single normalized model.

**Schemas (`evals/core/schemas.py`).** `EvalSample` with canonical fields `input`, `output`, `reference`, `contexts`, `source_id`, and `metadata` (defaults as in the current draft: optional strings, `contexts` defaults to `[]`, `metadata` to `{}`). Pydantic validation aliases: `user_input` → `input`, `response` → `output`, `retrieved_contexts` → `contexts`; serialized JSON uses canonical names only. `MetricResult`: required `name`; optional `value`, `reasoning`, and `error`.

**Datasets (`evals/core/datasets.py`).** Protocol `Dataset` with `async load() -> list[EvalSample]`. Implementations may set integer `total_items`; otherwise `Evaluator` uses `len(loaded samples)` for `EvaluationResponse.total_items`.

**Metrics (`evals/core/metrics.py`).** `SampleField` enum: `input`, `output`, `reference`, `contexts`. Abstract `Metric` with `name`, `description`, `required_fields: frozenset[SampleField]`, and `async score(sample) -> MetricResult`. Callable-backed `MetricDefinition(name, description, scorer, required_fields)` delegates scoring to an async `scorer(sample)`.

**Required-field gating lives on `Metric.score`, not in `Evaluator`.** Every evaluation path must call `await metric.score(sample)`—never the raw `scorer` directly from the evaluator loop. Before `_score` / the callable runs, treat a required field as missing when its value is `None`, `""`, or `[]`. If anything is missing: do **not** invoke the scorer; return `MetricResult(name=<metric name>, value=None, reasoning=None, error=<message>)` without raising. Error text is deterministic: sorted missing field **names**, comma-separated, then `" is required"` (one field) or `" are required"` (multiple).

**Evaluator (`evals/core/evaluator.py`).** `Evaluator(metrics: Sequence[Metric])` with `async evaluate(dataset) -> EvaluationResponse`. Load samples, then for each sample sequentially run every metric via `await metric.score(sample)` (no parallel `gather` required). Build:

- `EvaluationResponse(total_items, evaluated_items, duration_seconds, metrics, items)`
- `MetricMetadata(name, description, required_fields)` — `required_fields` sorted strings
- `EvaluationItemResult(item_id, sample, metrics)` — **`sample` is the full normalized `EvalSample`**, not a truncated dict; `item_id` is `sample.source_id` or `f"sample-{index}"`

`evaluate` returns only this response (no report file I/O).

**Runner + reporting.** Abstract `EvalRunner.run_eval(generate_report_file: bool = True) -> dict[str, Any]` in `evals/core/runner.py`. In `evals/core/reporting.py`, `build_report(response, run_type, evaluation_model, dataset_source) -> EvaluationReport` aggregates `EvaluationResponse` into a stable report (`schema_version`, timestamps, counts, `metrics_summary` with per-metric `average_score` over non-error results, item previews). `write_report(report)` persists under `evals/reports/` when runners opt in.

**Example (`evals/example_eval/`).** Inline `Dataset` with exactly three samples (`source_id` `example-1` … `example-3`). One deterministic `MetricDefinition` `exact_match` (requires `output` and `reference`; description `Whether output and reference are equal after normalization.`). Scoring compares `(output or "").strip().casefold()` to `(reference or "").strip().casefold()`; reasoning `Output matches the reference.` or `Output does not match the reference.`. `ExampleEvaluationRunner(EvalRunner)` calls `Evaluator.evaluate`, then `build_report` with `run_type="example_evaluation"`, `evaluation_model="deterministic"`, and `dataset_source` from the dataset’s `source_name` (`inline_example`). `run_eval(generate_report_file=False)` returns `report.model_dump(mode="json")` and must not write a file. Module entry point runs offline without secrets.

## Stack

- python

## Why this is hard

The most plausible first implementation copies **`evals/rag_evals/evaluator.py`**: keep `required_fields` on metrics for reporting, but implement gating in the **evaluator loop**—e.g. only skip scoring when reference is missing, or wrap `await metric.scorer(sample)` in try/except while calling the scorer directly. That matches how this repo already evaluates Ragas samples and passes happy-path examples where every field is populated.

It fails once samples use the **normalized** optional fields: a metric that requires `contexts` or `output` still runs the async scorer when `contexts` is `[]` or `output` is `""`, because the rag_evals gate never treated those as missing. Tests use a scorer spy and assert the callable is **never** invoked on invalid samples; a rag_evals-style loop leaves the spy counting calls while metadata looks correct.

## Verification

Run `tests/test.sh` offline with no network or secrets. Pytest focus: `tests/evals/test_eval_core.py` and `tests/evals/test_example_eval.py`.

**Metric gating on `Metric.score` (unit).** Async scorer spy with a call counter. `MetricDefinition` named `groundedness` requiring `contexts` and `output`. For each invalid form (`output=None`, `output=""`, `contexts=[]`), `await metric.score(...)` must not raise, must not increment the spy, and must return `MetricResult` with `value is None`, `reasoning is None`, and a non-null `error`. Both missing:

```json
{"name": "groundedness", "value": null, "reasoning": null, "error": "contexts, output are required"}
```

Only `output` missing: error exactly `"output is required"`. Valid sample with non-empty `output` and `contexts`: spy called once, scorer result returned. Alias fields normalize to canonical `EvalSample` on dump.

**Same gating through `Evaluator.evaluate` (catches the why-hard wrong fix).** The rag_evals-style mistake keeps `required_fields` for metadata but gates in the evaluator loop (reference-only skip, or `await metric.scorer(sample)` instead of `await metric.score(sample)`). Unit tests on `metric.score` alone still pass that implementation whenever the happy path calls the scorer with full fields.

Integration checks load samples through a mock `Dataset` and run `await Evaluator([metric]).evaluate(dataset)` with the same spy scorer and `groundedness` metric as above. The spy call count applies to the whole `evaluate` call.

| Sample shape (invalid field) | `reference` present | Why a rag_evals copy still fails |
|------------------------------|--------------------|----------------------------------|
| `output=""`, `contexts=["ctx"]` | yes | Reference gate passes; direct `scorer` runs unless `Metric.score` blocks empty `output`. |
| `output="ok"`, `contexts=[]` | yes | Same: reference present but `contexts` missing. |
| `output=None`, `contexts=[]` | yes | Both non-reference requirements missing; scorer must not run. |

For each row, assert: spy unchanged after `evaluate`; exactly one `EvaluationItemResult`; item metric error matches the same deterministic messages as the unit tests (e.g. `"output is required"` or `"contexts, output are required"`). Then a two-item dataset: first sample invalid (`contexts=[]`, populated `output`/`reference`), second sample valid for the metric—spy count must be **1** after one `evaluate`, proving the evaluator loop does not bypass `Metric.score` for the valid row while incorrectly scoring the invalid row.

**No evaluator-local reference shortcut.** One integration case uses a sample with `reference="present"`, `output=""`, and non-empty `contexts`. A wrong implementation that only skips scoring when reference is missing (mirroring `RagEvaluator._evaluate_sample`) must still leave the spy at zero and return an `output is required` error on the item—not a successful score and not a raised exception.

**Example runner.** `await ExampleEvaluationRunner().evaluator.evaluate(dataset)` serialized with `model_dump(mode="json")`: `total_items == evaluated_items == 3`, non-negative `duration_seconds`, metric metadata for `exact_match` as specified, items `example-1` … `example-3` each with full canonical `sample` keys and metric values `1.0`, `0.0`, `1.0` in order with the stated reasoning strings. `await run_eval(generate_report_file=False)`: no file written; `run_type == "example_evaluation"`, `dataset_source == "inline_example"`, `evaluated_items == 3`, `metrics_summary["exact_match"]["average_score"]` ≈ `2/3`.

## Upstream

| Field | Value |
|-------|-------|
| Repository | `apssouza22/production-rag` |
| Base commit | `f030fe10c1ad1c293eee431b774c80bdf75112ea` |

## Relevance to expertise

I maintain this RAG stack and its eval tooling; I added the existing `custom_evals` and `rag_evals` runners and need a shared core so new metrics share one validation and reporting path instead of duplicating rag_evals-style loops.

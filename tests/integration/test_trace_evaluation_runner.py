import json
from functools import partial
from unittest.mock import AsyncMock, Mock, patch

import pytest
from evals.core.reporting import write_report
from evals.live_req_evals.config import Settings
from evals.live_req_evals.metrics import load_metric_prompts
from evals.live_req_evals.runner import TraceEvaluationRunner


@pytest.mark.parametrize("generate_report", [False, True])
async def test_trace_evaluation_runner_with_mocked_external_services(tmp_path, monkeypatch, generate_report: bool) -> None:
    settings = Settings(
        evaluation_llm="mock-judge",
        evaluation_api_key="openai-test-key",
        evaluation_sleep_time=0,
        langfuse_public_key="langfuse-test-public-key",
        langfuse_secret_key="langfuse-test-secret-key",
        langfuse_host="https://langfuse.test",
    )
    trace = Mock(
        id="trace-1",
        input={"query": "What is retrieval-augmented generation?"},
        output={"answer": "It combines retrieval with language-model generation."},
        scores=[],
    )

    langfuse = Mock()
    langfuse.api.trace.list.return_value.data = [trace]

    judge = Mock()
    judge.ainvoke = AsyncMock(return_value={"score": 0.9, "reasoning": "The answer is relevant and supported."})
    chat_model = Mock()
    chat_model.with_structured_output.return_value = judge

    monkeypatch.setattr(
        "evals.live_req_evals.runner.write_report",
        partial(write_report, reports_dir=tmp_path),
    )

    with (
        patch("evals.live_req_evals.runner.get_settings", return_value=settings),
        patch("evals.live_req_evals.runner.Langfuse", return_value=langfuse) as langfuse_cls,
        patch("evals.core.llm.ChatOpenAI", return_value=chat_model) as openai_cls,
    ):
        runner = TraceEvaluationRunner()
        report = await runner.run_eval(generate_report_file=generate_report)

    metric_names = set(load_metric_prompts())

    langfuse_cls.assert_called_once_with(
        public_key="langfuse-test-public-key",
        secret_key="langfuse-test-secret-key",
        host="https://langfuse.test",
        timeout=60,
    )
    openai_cls.assert_called_once_with(
        model="mock-judge",
        api_key="openai-test-key",
        base_url="https://api.openai.com/v1",
    )
    langfuse.api.trace.list.assert_called_once()
    assert judge.ainvoke.await_count == len(metric_names)
    assert langfuse.create_score.call_count == len(metric_names)
    assert {call.kwargs["name"] for call in langfuse.create_score.call_args_list} == metric_names

    assert report["run_type"] == "trace_evaluation"
    assert report["dataset_source"] == "langfuse_traces"
    assert report["total_items"] == 1
    assert report["evaluated_items"] == 1
    assert report["successful_items"] == 1
    assert report["failed_items"] == 0
    assert set(report["metrics_summary"]) == metric_names
    assert all(summary["average_score"] == 0.9 for summary in report["metrics_summary"].values())

    if generate_report:
        report_path = tmp_path / str(report["report_path"]).split("/")[-1]
        assert report_path.exists()
        assert json.loads(report_path.read_text(encoding="utf-8"))["run_type"] == "trace_evaluation"
    else:
        assert report["report_path"] is None

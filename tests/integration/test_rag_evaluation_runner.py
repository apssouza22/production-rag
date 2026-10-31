import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from evals.rag_evals.config import RagEvalSettings
from evals.rag_evals.runner import RagEvaluationRunner
from pydantic import BaseModel


def _structured_response(response_model: type[BaseModel]) -> BaseModel:
    payloads = {
        "ClaimDecompositionOutput": {"claims": ["RAG combines retrieval with generation."]},
        "ContextPrecisionOutput": {"reason": "The context supports the answer.", "verdict": 1},
        "ContextRecallOutput": {
            "classifications": [
                {
                    "statement": "RAG combines retrieval with generation.",
                    "reason": "The statement is present in the context.",
                    "attributed": 1,
                }
            ]
        },
        "ContextRelevanceOutput": {"rating": 2},
        "NLIStatementOutput": {
            "statements": [
                {
                    "statement": "RAG combines retrieval with generation.",
                    "reason": "The context supports the statement.",
                    "verdict": 1,
                }
            ]
        },
        "StatementGeneratorOutput": {"statements": ["RAG combines retrieval with generation."]},
    }
    return response_model.model_validate(payloads[response_model.__name__])


@pytest.mark.parametrize("generate_report", [False, True])
async def test_rag_evaluation_runner_with_mocked_external_services(tmp_path: Path, generate_report: bool) -> None:
    dataset_path = tmp_path / "queries.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "user_input": "What is RAG?",
                    "reference": "RAG combines retrieval with generation.",
                    "response": "RAG combines retrieval with generation.",
                }
            ]
        ),
        encoding="utf-8",
    )
    settings = RagEvalSettings(
        evaluation_llm="mock-judge",
        evaluation_api_key="openai-test-key",
        rag_api_base_url="https://rag-api.test",
        rag_eval_dataset_path=dataset_path,
        rag_eval_top_k=1,
        rag_eval_request_timeout=5,
    )

    api_response = Mock()
    api_response.raise_for_status = Mock()
    api_response.json.return_value = {
        "query": "What is RAG?",
        "rerank_applied": True,
        "after_rerank": [
            {
                "chunk_text": "RAG combines retrieval with language-model generation.",
                "arxiv_id": "test-paper",
                "title": "RAG",
                "rank": 1,
            }
        ],
    }
    http_client = Mock()
    http_client.post = AsyncMock(return_value=api_response)
    http_client_context = MagicMock()
    http_client_context.__aenter__ = AsyncMock(return_value=http_client)
    http_client_context.__aexit__ = AsyncMock(return_value=None)

    judges: list[Mock] = []
    chat_model = Mock()

    def make_structured_judge(response_model: type[BaseModel]) -> Mock:
        judge = Mock()
        judge.ainvoke = AsyncMock(return_value=_structured_response(response_model))
        judges.append(judge)
        return judge

    chat_model.with_structured_output.side_effect = make_structured_judge
    reports_dir = tmp_path / "reports"

    with (
        patch("evals.core.llm.ChatOpenAI", return_value=chat_model) as openai_cls,
        patch("evals.rag_evals.retrieval.httpx.AsyncClient", return_value=http_client_context) as http_client_cls,
        patch("evals.rag_evals.runner.REPORTS_DIR", reports_dir),
    ):
        runner = RagEvaluationRunner(settings=settings)
        report = await runner.run_eval(generate_report_file=generate_report)

    openai_cls.assert_called_once_with(
        model="mock-judge",
        api_key="openai-test-key",
        base_url="https://api.openai.com/v1",
    )
    http_client_cls.assert_called_once_with(timeout=5)
    http_client.post.assert_awaited_once_with(
        "https://rag-api.test/api/v1/rerank-search",
        json={"query": "What is RAG?", "top_k": 1, "use_hybrid": True, "rerank_enabled": True},
    )
    api_response.raise_for_status.assert_called_once_with()
    assert judges
    assert all(judge.ainvoke.await_count == 1 for judge in judges)

    expected_metrics = {
        "context_precision",
        "context_recall",
        "context_relevance",
        "context_utilization",
        "factual_correctness",
        "faithfulness",
    }
    assert report["run_type"] == "rag_evaluation"
    assert report["dataset_source"] == "json_rag"
    assert report["total_items"] == 1
    assert report["evaluated_items"] == 1
    assert report["successful_items"] == 1
    assert report["failed_items"] == 0
    assert set(report["metrics_summary"]) == expected_metrics
    assert all(summary["average_score"] == 1.0 for summary in report["metrics_summary"].values())

    if generate_report:
        report_path = Path(report["report_path"])
        assert report_path.parent == reports_dir
        assert json.loads(report_path.read_text(encoding="utf-8"))["run_type"] == "rag_evaluation"
    else:
        assert report["report_path"] is None

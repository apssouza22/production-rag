"""Tests for trace input/output extraction."""

from types import SimpleNamespace

from rag_evals.helpers import get_input_output


def test_extracts_rag_query_and_answer():
    trace = SimpleNamespace(
        input={"query": "What is RAG?"},
        output={"answer": "Retrieval augmented generation combines search with LLMs."},
    )

    input_text, output_text = get_input_output(trace)

    assert input_text == "What is RAG?"
    assert output_text == "Retrieval augmented generation combines search with LLMs."


def test_extracts_langgraph_messages_from_output():
    trace = SimpleNamespace(
        input=None,
        output={
            "messages": [
                {"type": "human", "content": "Hello"},
                {"type": "ai", "content": "Hi there"},
            ]
        },
    )

    input_text, output_text = get_input_output(trace)

    assert input_text == "human: Hello"
    assert output_text == "ai: Hi there"


def test_returns_none_for_missing_payloads():
    trace = SimpleNamespace(input=None, output=None)

    input_text, output_text = get_input_output(trace)

    assert input_text is None
    assert output_text is None

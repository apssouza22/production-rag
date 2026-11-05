import pytest
from src.domain.rrf import QueryBuilder


def test_query_builder_basic_query():
    builder = QueryBuilder(query="machine learning", size=5)

    query = builder.build()

    assert query["size"] == 5
    assert query["from"] == 0
    assert query["track_total_hits"] is True

    bool_query = query["query"]["bool"]
    assert len(bool_query["must"]) == 1

    multi_match = bool_query["must"][0]["multi_match"]
    assert multi_match["query"] == "machine learning"
    assert "contextualized_text^4" in multi_match["fields"]
    assert "chunk_text^2" in multi_match["fields"]


def test_query_builder_with_ingestion_name():
    builder = QueryBuilder(query="rag eval", ingestion_name="evals-local-pdfs")

    query = builder.build()

    filters = query["query"]["bool"]["filter"]
    assert {"term": {"ingestion_name": "evals-local-pdfs"}} in filters


def test_query_builder_with_categories():
    builder = QueryBuilder(query="deep learning", categories=["cs.AI", "cs.LG"])

    query = builder.build()

    bool_query = query["query"]["bool"]
    assert "filter" in bool_query

    filters = bool_query["filter"]
    assert len(filters) == 1
    assert filters[0]["terms"]["categories"] == ["cs.AI", "cs.LG"]


def test_query_builder_latest_papers_sorting():
    builder = QueryBuilder(query="neural networks", latest_papers=True)

    query = builder.build()

    assert "sort" in query
    sort_config = query["sort"]
    assert len(sort_config) == 2
    assert sort_config[0]["updated_at"]["order"] == "desc"
    assert sort_config[1] == "_score"


def test_query_builder_relevance_sorting():
    builder = QueryBuilder(query="transformers attention", latest_papers=False)

    query = builder.build()

    assert "sort" not in query


def test_query_builder_empty_query_sorting():
    builder = QueryBuilder(query="", latest_papers=False)

    query = builder.build()

    assert "sort" in query
    sort_config = query["sort"]
    assert sort_config[0]["updated_at"]["order"] == "desc"


def test_query_builder_highlighting():
    builder = QueryBuilder(query="test query")

    query = builder.build()

    highlight = query["highlight"]
    assert "fields" in highlight

    fields = highlight["fields"]
    assert "contextualized_text" in fields
    assert "chunk_text" in fields
    assert fields["contextualized_text"]["fragment_size"] == 150
    assert fields["chunk_text"]["pre_tags"] == ["<mark>"]


def test_query_builder_source_fields():
    builder = QueryBuilder(query="test query")

    query = builder.build()

    assert query["_source"] == {"excludes": ["embedding"]}


def test_query_builder_custom_fields():
    custom_fields = ["section_title^5", "chunk_text^1"]
    builder = QueryBuilder(query="test", fields=custom_fields)

    query = builder.build()

    multi_match = query["query"]["bool"]["must"][0]["multi_match"]
    assert multi_match["fields"] == custom_fields

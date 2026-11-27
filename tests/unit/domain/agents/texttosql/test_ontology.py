"""Tests for schema ontology loading and prompt formatting."""

from pathlib import Path

import pytest
from src.agents.texttosql.ontology import load_schema_ontology
from src.agents.texttosql.prompts import GENERATE_QUERY_SYSTEM_PROMPT

FIXTURE_PATH = Path(__file__).resolve().parents[4] / "fixtures" / "texttosql" / "mini-ontology.yaml"
DEFAULT_ONTOLOGY = Path(__file__).resolve().parents[5] / "src" / "agents" / "texttosql" / "pagila-ontology.yaml"


class TestSchemaOntology:
    def test_load_mini_ontology_table_names(self):
        ontology = load_schema_ontology(FIXTURE_PATH)
        assert ontology.table_names() == ["film", "customer"]

    def test_format_for_prompt_includes_sections_and_sql(self):
        ontology = load_schema_ontology(FIXTURE_PATH)
        text = ontology.format_for_prompt()
        assert "### table: `film`" in text
        assert "rental_by_customer" in text
        assert "customer_total_payments" in text
        assert "SELECT customer_id, SUM(amount)" in text

    def test_load_default_pagila_ontology_has_fifteen_tables(self):
        if not DEFAULT_ONTOLOGY.is_file():
            pytest.skip("pagila-ontology.yaml not present")
        ontology = load_schema_ontology(DEFAULT_ONTOLOGY)
        assert len(ontology.table_names()) == 15
        assert "actor" in ontology.table_names()

    def test_generate_prompt_has_no_arxiv_papers_blurb(self):
        ontology = load_schema_ontology(FIXTURE_PATH)
        prompt = GENERATE_QUERY_SYSTEM_PROMPT.format(
            dialect="postgresql",
            top_k=5,
            ontology=ontology.format_for_prompt(),
        )
        assert "arxiv" not in prompt.lower()
        assert "DVD catalog title." in prompt
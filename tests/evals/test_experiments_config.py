"""Tests for experiment JSON settings loading."""

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest
from evals.experiments.config import (
    ExperimentSettings,
    QuerySettings,
    apply_experiment_platform_settings,
    build_run_config,
    load_experiment_config,
    resolve_settings_path,
    sync_experiment_opensearch_partition,
)
from src.config import PDFParserSettings


def test_resolve_settings_path_relative_to_repo_root() -> None:
    resolved = resolve_settings_path("evals/experiments/data/settings/experiment-config.json")
    assert resolved.is_absolute()
    assert resolved.name == "experiment-config.json"
    assert "evals/experiments/data/settings" in str(resolved)


def test_resolve_settings_path_absolute_unchanged(tmp_path: Path) -> None:
    absolute = tmp_path / "custom.json"
    assert resolve_settings_path(absolute) == absolute


def test_load_experiment_config_merges_overrides(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment-config.json"
    config_path.write_text(
        json.dumps(
            {
                "pdf_parser": {"max_pages": 12, "do_ocr": True},
                "query": {"rerank_enabled": False, "rerank_candidate_multiplier": 4},
                "experiment": {"experiment_top_k": 7, "experiment_ingestion_name": "json-partition"},
            }
        ),
        encoding="utf-8",
    )

    loaded = load_experiment_config(config_path)

    assert loaded.pdf_parser.max_pages == 12
    assert loaded.pdf_parser.do_ocr is True
    assert loaded.pdf_parser.max_file_size_mb == 20
    assert loaded.query.rerank_enabled is False
    assert loaded.query.rerank_candidate_multiplier == 4
    assert loaded.experiment.experiment_top_k == 7
    assert loaded.experiment.experiment_ingestion_name == "json-partition"


def test_load_experiment_config_rejects_invalid_json(tmp_path: Path) -> None:
    config_path = tmp_path / "bad.json"
    config_path.write_text("{not json", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid JSON"):
        load_experiment_config(config_path)


def test_apply_experiment_platform_settings_sets_env_and_clears_caches() -> None:
    pdf_parser = PDFParserSettings.model_validate({"max_pages": 11, "do_ocr": True})
    chunking = load_experiment_config().chunking

    with (
        patch("evals.experiments.config.make_pdf_parser_service.cache_clear") as clear_parser,
        patch("evals.experiments.config.get_cached_services.cache_clear") as clear_services,
    ):
        apply_experiment_platform_settings(pdf_parser, chunking)

    assert os.environ["PDF_PARSER__MAX_PAGES"] == "11"
    assert os.environ["PDF_PARSER__DO_OCR"] == "true"
    clear_parser.assert_called_once()
    clear_services.assert_called_once()


def test_query_settings_to_rerank_config() -> None:
    query = QuerySettings(use_hybrid=False, rerank_enabled=False, rerank_candidate_multiplier=3)
    config = query.to_rerank_config()

    assert config.use_hybrid is False
    assert config.rerank_enabled is False
    assert config.rerank_candidate_multiplier == 3


def test_build_run_config_syncs_opensearch_ingestion_partition(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "experiment-config.json"
    config_path.write_text(
        json.dumps({"experiment": {"experiment_ingestion_name": "eval-partition"}}),
        encoding="utf-8",
    )
    monkeypatch.delenv("OPENSEARCH__INGESTION_NAME", raising=False)

    build_run_config(config_path)

    assert os.environ["OPENSEARCH__INGESTION_NAME"] == "eval-partition"


def test_build_run_config_applies_json_before_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config_path = tmp_path / "experiment-config.json"
    config_path.write_text(
        json.dumps({"experiment": {"experiment_ingestion_name": "from-json"}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("EXPERIMENT_INGESTION_NAME", "from-env")

    run_config = build_run_config(config_path)

    assert run_config.experiment.experiment_ingestion_name == "from-env"
    assert run_config.config_path == config_path


def test_load_experiment_config_parses_retrieval_mode(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment-config.json"
    config_path.write_text(json.dumps({"experiment": {"retrieval_mode": "kg"}}), encoding="utf-8")

    loaded = load_experiment_config(config_path)

    assert loaded.experiment.retrieval_mode == "kg"


def test_sync_experiment_partition_sets_kg_ingestion_for_kg_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("KG__INGESTION_NAME", raising=False)
    experiment = ExperimentSettings(experiment_ingestion_name="eval-partition", retrieval_mode="kg")

    sync_experiment_opensearch_partition(experiment)

    assert os.environ["OPENSEARCH__INGESTION_NAME"] == "eval-partition"
    assert os.environ["KG__INGESTION_NAME"] == "eval-partition"

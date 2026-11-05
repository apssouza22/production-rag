"""Tests for experiment runner retrieval wiring."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from evals.experiments.config import ExperimentRunConfig, ExperimentSettings, QuerySettings
from evals.experiments.runner import ExperimentRunner
from src.config import ChunkingSettings, PDFParserSettings


def test_runner_rejects_kg_mode_when_kg_disabled() -> None:
    run_config = ExperimentRunConfig(
        pdf_parser=PDFParserSettings(),
        chunking=ChunkingSettings(),
        query=QuerySettings(),
        experiment=ExperimentSettings(retrieval_mode="kg"),
        config_path=Path("evals/experiments/data/settings/experiment-config.json"),
    )
    app_settings = Mock()
    app_settings.kg.enabled = False

    with (
        patch("evals.experiments.runner.get_app_settings", return_value=app_settings),
        pytest.raises(RuntimeError, match="retrieval_mode=kg"),
    ):
        ExperimentRunner(run_config=run_config, metrics=[])

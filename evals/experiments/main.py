#!/usr/bin/env python3
"""CLI for the local-PDF retrieval experiment."""

import argparse
import asyncio
import logging
import os
import sys

import colorama
from colorama import Fore, Style

from evals.core.report_publisher import make_report_publisher
from evals.experiments.config import (
    DEFAULT_EXPERIMENT_CONFIG_PATH,
    REPO_ROOT,
    ExperimentRunConfig,
    build_run_config,
    resolve_settings_path,
)
from evals.experiments.runner import REPORTS_DIR, ExperimentRunner


def print_title(title: str) -> None:
    print("\n" + "=" * 60)
    print(f"{Fore.CYAN}{Style.BRIGHT}{title.center(60)}{Style.RESET_ALL}")
    print("=" * 60 + "\n")


def print_info(message: str) -> None:
    print(f"{Fore.GREEN}• {message}{Style.RESET_ALL}")


def print_error(message: str) -> None:
    print(f"{Fore.RED}✗ {message}{Style.RESET_ALL}")


def print_success(message: str) -> None:
    print(f"{Fore.GREEN}✓ {message}{Style.RESET_ALL}")


def display_summary(report: dict) -> None:
    print_title("Experiment Summary")
    print(f"{Fore.CYAN}Model:{Style.RESET_ALL} {report['evaluation_model']}")
    dataset = report["dataset_location"] or report["dataset_source"]
    print(f"{Fore.CYAN}Dataset:{Style.RESET_ALL} {dataset}")
    print(f"{Fore.CYAN}Items:{Style.RESET_ALL} {report['evaluated_items']}/{report['total_items']}")
    print(f"{Fore.CYAN}Duration:{Style.RESET_ALL} {report['duration_seconds']} seconds")

    ingestion = report["metadata"].get("ingestion", {})
    state = "skipped (already indexed)" if ingestion.get("skipped") else "performed"
    print(f"{Fore.CYAN}Ingestion:{Style.RESET_ALL} {state}, {ingestion.get('chunk_count', 0)} chunks")

    print("\n" + f"{Fore.CYAN}Metrics:{Style.RESET_ALL}")
    for metric_name, summary in report["metrics_summary"].items():
        total = summary["success_count"] + summary["failure_count"]
        if total == 0:
            print(f"  • {metric_name}: no scores")
        else:
            print(f"  • {metric_name}: avg={summary['average_score']:.4f}, success={summary['success_count']}/{total}")

    if report.get("report_path"):
        print(f"\n{Fore.CYAN}Report:{Style.RESET_ALL} {report['report_path']}")


async def run_experiment(args: argparse.Namespace, run_config: ExperimentRunConfig) -> None:
    settings = run_config.experiment
    print_title("Starting Local-PDF Retrieval Experiment")
    print_info(f"Settings: {run_config.config_path}")
    print_info(f"PDFs: {settings.experiment_pdfs_dir}")
    print_info(f"Ingestion partition: {settings.experiment_ingestion_name}")
    print_info(f"Judge model: {settings.evaluation_llm}")

    try:
        runner = ExperimentRunner(
            run_config=run_config,
            force_ingest=args.force_ingest,
            regenerate_queries=args.regenerate_queries,
            report_publisher=make_report_publisher(enabled=not args.no_report, reports_dir=REPORTS_DIR),
        )
        report = await runner.run_eval()
        print_success("Experiment evaluation completed successfully!")
        display_summary(report)
    except Exception as exc:
        print_error(f"Experiment evaluation failed: {exc}")
        logging.exception("Experiment evaluation failed")
        sys.exit(1)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    colorama.init()

    default_settings = str(DEFAULT_EXPERIMENT_CONFIG_PATH.relative_to(REPO_ROOT))

    parser = argparse.ArgumentParser(description="Ingest local arXiv PDFs and evaluate retrieval with Ragas")
    parser.add_argument(
        "--settings",
        default=default_settings,
        help="Path to experiment JSON settings (default: evals/experiments/data/settings/experiment-config.json)",
    )
    parser.add_argument("--ingestion-name", help="Override EXPERIMENT_INGESTION_NAME for this run")
    parser.add_argument("--force-ingest", action="store_true", help="Re-run ingestion even if the partition has chunks")
    parser.add_argument("--regenerate-queries", action="store_true", help="Regenerate the cached question dataset")
    parser.add_argument("--no-report", action="store_true", help="Skip writing a JSON report")
    args = parser.parse_args()

    if args.ingestion_name:
        os.environ["EXPERIMENT_INGESTION_NAME"] = args.ingestion_name

    run_config = build_run_config(resolve_settings_path(args.settings))

    # OpenSearch filters every search by the configured partition, so indexing and
    # retrieval must agree on the same ingestion name.
    os.environ["OPENSEARCH__INGESTION_NAME"] = run_config.experiment.experiment_ingestion_name

    asyncio.run(run_experiment(args, run_config))


if __name__ == "__main__":
    main()

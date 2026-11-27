#!/usr/bin/env python3
"""CLI for text-to-SQL execution-accuracy evaluation."""

import argparse
import asyncio
import logging
import sys

import colorama
from colorama import Fore, Style

from evals.core.report_publisher import make_report_publisher
from evals.texttosql_evals.config import get_settings
from evals.texttosql_evals.runner import REPORTS_DIR, TextToSQLEvaluationRunner

settings = get_settings()


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
    print_title("Evaluation Summary")
    print(f"{Fore.CYAN}Model:{Style.RESET_ALL} {report['evaluation_model']}")
    dataset = report["dataset_location"] or report["dataset_source"]
    print(f"{Fore.CYAN}Dataset:{Style.RESET_ALL} {dataset}")
    print(f"{Fore.CYAN}Items:{Style.RESET_ALL} {report['evaluated_items']}/{report['total_items']}")
    print(f"{Fore.CYAN}Duration:{Style.RESET_ALL} {report['duration_seconds']} seconds")

    print("\n" + f"{Fore.CYAN}Metrics:{Style.RESET_ALL}")
    for metric_name, summary in report["metrics_summary"].items():
        total = summary["success_count"] + summary["failure_count"]
        description = summary.get("description", "")
        if total == 0:
            print(f"  • {metric_name}: no scores")
        else:
            print(f"  • {metric_name}: avg={summary['average_score']:.4f}, success={summary['success_count']}/{total}")
        if description:
            print(f"    {description}")

    if report.get("report_path"):
        print(f"\n{Fore.CYAN}Report:{Style.RESET_ALL} {report['report_path']}")


async def run_evaluation(write_report: bool = True) -> None:
    print_title("Starting Text-to-SQL Evaluation")
    print_info(f"Dataset: {settings.text_to_sql_eval_dataset_path}")

    try:
        runner = TextToSQLEvaluationRunner(
            settings=settings,
            report_publisher=make_report_publisher(enabled=write_report, reports_dir=REPORTS_DIR),
        )
        report = await runner.run_eval()
        print_success("Text-to-SQL evaluation completed successfully!")
        display_summary(report)
    except Exception as exc:
        print_error(f"Text-to-SQL evaluation failed: {exc}")
        logging.exception("Text-to-SQL evaluation failed")
        sys.exit(1)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    colorama.init()

    parser = argparse.ArgumentParser(description="Run text-to-SQL execution-accuracy evaluation")
    parser.add_argument("--no-report", action="store_true", help="Skip writing a JSON report")
    parser.add_argument("--quick", action="store_true", help="Run with default settings")
    args = parser.parse_args()

    asyncio.run(run_evaluation(write_report=not args.no_report))


if __name__ == "__main__":
    main()
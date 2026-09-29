#!/usr/bin/env python3
"""CLI for Ragas-based retrieval evaluation."""

import argparse
import asyncio
import logging
import sys

import colorama
from colorama import Fore, Style

from evals.rag_evals.config import get_settings
from evals.rag_evals.evaluator import RagEvaluator
from evals.rag_evals.langfuse_experiment import RagLangfuseExperiment

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
    print_title("Ragas Evaluation Summary")
    print(f"{Fore.CYAN}Model:{Style.RESET_ALL} {report['evaluation_llm']}")
    print(f"{Fore.CYAN}Dataset:{Style.RESET_ALL} {report['dataset_path']}")
    print(f"{Fore.CYAN}Samples:{Style.RESET_ALL} {report['evaluated_samples']}/{report['total_queries']}")
    print(f"{Fore.CYAN}Duration:{Style.RESET_ALL} {report['duration_seconds']} seconds")

    print("\n" + f"{Fore.CYAN}Metrics:{Style.RESET_ALL}")
    for metric_name, summary in report["metrics_summary"].items():
        total = summary["success_count"] + summary["failure_count"]
        description = summary.get("description", "")
        if total == 0:
            print(f"  • {metric_name}: no scores")
        else:
            print(
                f"  • {metric_name}: avg={summary['avg_score']:.4f}, "
                f"success={summary['success_count']}/{total}"
            )
        if description:
            print(f"    {description}")

    if report.get("report_path"):
        print(f"\n{Fore.CYAN}Report:{Style.RESET_ALL} {report['report_path']}")


def display_langfuse_summary(result) -> None:
    print_title("Langfuse Experiment Summary")
    print(f"{Fore.CYAN}Experiment:{Style.RESET_ALL} {result.name}")
    print(f"{Fore.CYAN}Run:{Style.RESET_ALL} {result.run_name}")
    print(f"{Fore.CYAN}Items:{Style.RESET_ALL} {len(result.item_results)}")
    if result.dataset_run_url:
        print(f"{Fore.CYAN}Dataset run:{Style.RESET_ALL} {result.dataset_run_url}")
    print("\n" + result.format())


def run_langfuse_experiment(
    generate_report: bool = True,
    experiment_name: str | None = None,
    run_name: str | None = None,
    dataset_name: str | None = None,
) -> None:
    print_title("Starting Langfuse Experiment")
    print_info(f"RAG API: {settings.rag_api_base_url}")
    print_info(f"Dataset: {settings.rag_eval_langfuse_dataset or settings.rag_eval_dataset_path}")
    print_info(f"Judge model: {settings.evaluation_llm}")

    try:
        runner = RagLangfuseExperiment(settings=settings)
        result = runner.run(
            experiment_name=experiment_name,
            run_name=run_name,
            dataset_name=dataset_name,
            generate_report_file=generate_report,
        )
        print_success("Langfuse experiment completed successfully!")
        display_langfuse_summary(result)
    except Exception as exc:
        print_error(f"Langfuse experiment failed: {exc}")
        logging.exception("Langfuse experiment failed")
        sys.exit(1)


async def run_evaluation(generate_report: bool = True) -> None:
    print_title("Starting Ragas Evaluation")
    print_info(f"RAG API: {settings.rag_api_base_url}")
    print_info(f"Dataset: {settings.rag_eval_dataset_path}")
    print_info(f"Judge model: {settings.evaluation_llm}")

    try:
        evaluator = RagEvaluator(settings=settings)
        report = await evaluator.run(generate_report_file=generate_report)
        print_success("Ragas evaluation completed successfully!")
        display_summary(report)
    except Exception as exc:
        print_error(f"Ragas evaluation failed: {exc}")
        logging.exception("Ragas evaluation failed")
        sys.exit(1)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    colorama.init()

    parser = argparse.ArgumentParser(description="Run Ragas retrieval evaluation using rerank search contexts")
    parser.add_argument("--no-report", action="store_true", help="Skip writing a JSON report")
    parser.add_argument("--quick", action="store_true", help="Run with default settings")
    parser.add_argument(
        "--langfuse-experiment",
        action="store_true",
        help="Run as a Langfuse experiment (traces + scores in Langfuse UI)",
    )
    parser.add_argument("--experiment-name", help="Override RAG_EVAL_EXPERIMENT_NAME")
    parser.add_argument("--run-name", help="Optional Langfuse experiment run name")
    parser.add_argument(
        "--dataset-name",
        help="Use a hosted Langfuse dataset instead of the local JSON file (RAG_EVAL_LANGFUSE_DATASET)",
    )
    args = parser.parse_args()

    if args.langfuse_experiment:
        run_langfuse_experiment(
            generate_report=not args.no_report,
            experiment_name=args.experiment_name,
            run_name=args.run_name,
            dataset_name=args.dataset_name,
        )
        return

    asyncio.run(run_evaluation(generate_report=not args.no_report))


if __name__ == "__main__":
    main()

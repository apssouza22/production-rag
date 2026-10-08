"""Airflow task adapters for the arXiv ingestion domain."""

import logging
from datetime import datetime, timedelta

from src.domain.arxiv_ingestion.fetching import download_papers as fetch_papers
from src.domain.arxiv_ingestion.fetching import parse_pdfs as parse_downloaded_pdfs
from src.domain.arxiv_ingestion.indexing import index_papers_hybrid as index_papers
from src.domain.arxiv_ingestion.reporting import generate_daily_report as build_daily_report

logger = logging.getLogger(__name__)


def _resolve_target_date(context: dict) -> str:
    """Resolve the target ingestion date from Airflow execution context."""
    execution_date = context.get("execution_date")
    target_date = execution_date - timedelta(days=1) if execution_date else datetime.now() - timedelta(days=1)
    return target_date.strftime("%Y%m%d")


def download_papers(**context) -> dict:
    """Fetch papers and publish download results through Airflow XCom."""
    target_date = _resolve_target_date(context)
    results = fetch_papers(target_date)

    ti = context.get("ti")
    if ti:
        ti.xcom_push(key="download_results", value=results)
    return results


def parse_pdfs(**context) -> dict:
    """Parse downloaded papers, reading and publishing results through XCom."""
    ti = context.get("ti")
    download_results = ti.xcom_pull(task_ids="download_papers", key="download_results") or {} if ti else {}
    results = parse_downloaded_pdfs(download_results)
    if ti:
        ti.xcom_push(key="parse_results", value=results)
    return results


def index_papers_hybrid(**context) -> dict:
    """Index papers and publish indexing results through Airflow XCom."""
    ti = context.get("ti")
    parse_results = ti.xcom_pull(task_ids="parse_pdfs", key="parse_results") if ti else None
    stats = index_papers(parse_results)
    if ti:
        ti.xcom_push(key="hybrid_index_stats", value=stats)
    return stats


def generate_daily_report(**context) -> dict:
    """Collect prior task results through XCom and publish the report."""
    ti = context.get("ti")
    if not ti:
        logger.warning("No task instance available, generating report with empty task results")

    download_stats = ti.xcom_pull(task_ids="download_papers", key="download_results") or {} if ti else {}
    parse_stats = ti.xcom_pull(task_ids="parse_pdfs", key="parse_results") or {} if ti else {}
    hybrid_stats = ti.xcom_pull(task_ids="index_papers_hybrid", key="hybrid_index_stats") or {} if ti else {}
    report = build_daily_report(
        download_stats,
        parse_stats,
        hybrid_stats,
        execution_date=context.get("execution_date"),
    )
    if ti:
        ti.xcom_push(key="daily_report", value=report)
    return report

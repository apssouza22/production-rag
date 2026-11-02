"""Airflow task adapters for the arXiv ingestion domain."""

import logging
from datetime import datetime, timedelta

from src.domain.arxiv_ingestion.reporting import generate_daily_report as build_daily_report
from src.domain.arxiv_ingestion.schemas import DownloadResults, PersistResults
from src.domain.arxiv_ingestion.service import get_ingestion_service

logger = logging.getLogger(__name__)


def _resolve_target_date(context: dict) -> str:
    """Resolve the target ingestion date from Airflow execution context."""
    execution_date = context.get("execution_date")
    target_date = execution_date - timedelta(days=1) if execution_date else datetime.now() - timedelta(days=1)
    return target_date.strftime("%Y%m%d")


def download_papers(**context) -> dict:
    """Fetch papers and publish download results through Airflow XCom."""
    target_date = _resolve_target_date(context)
    results = get_ingestion_service().download_papers(target_date)
    payload = results.model_dump(mode="json")

    ti = context.get("ti")
    if ti:
        ti.xcom_push(key="download_results", value=payload)
    return payload


def parse_pdfs_to_markdown(**context) -> dict:
    """Parse downloaded PDFs to shared Markdown artifacts."""
    ti = context.get("ti")
    download_results = DownloadResults.from_xcom(
        ti.xcom_pull(task_ids="download_papers", key="download_results") if ti else None,
    )
    if download_results is None:
        download_results = DownloadResults()
    results = get_ingestion_service().parse_pdfs_to_markdown(download_results)
    if ti:
        ti.xcom_push(key="markdown_results", value=results)
    return results


def persist_papers(**context) -> dict:
    """Persist downloaded paper metadata and parsed artifacts."""
    ti = context.get("ti")
    download_results = DownloadResults.from_xcom(
        ti.xcom_pull(task_ids="download_papers", key="download_results") if ti else None,
    )
    if download_results is None:
        download_results = DownloadResults()
    markdown_results = ti.xcom_pull(task_ids="parse_pdfs_to_markdown", key="markdown_results") or {} if ti else {}
    results = get_ingestion_service().persist_papers(download_results, markdown_results)
    payload = results.model_dump(mode="json")
    if ti:
        ti.xcom_push(key="persist_results", value=payload)
    return payload


def index_papers_hybrid(**context) -> dict:
    """Index papers and publish indexing results through Airflow XCom."""
    ti = context.get("ti")
    persist_results = PersistResults.from_xcom(
        ti.xcom_pull(task_ids="persist_papers", key="persist_results") if ti else None,
    )
    stats = get_ingestion_service().index_papers_hybrid(persist_results)
    if ti:
        ti.xcom_push(key="hybrid_index_stats", value=stats)
    return stats


def generate_daily_report(**context) -> dict:
    """Collect prior task results through XCom and publish the report."""
    ti = context.get("ti")
    if not ti:
        logger.warning("No task instance available, generating report with empty task results")

    download_stats = DownloadResults.from_xcom(
        ti.xcom_pull(task_ids="download_papers", key="download_results") if ti else None,
    )
    persist_stats = PersistResults.from_xcom(
        ti.xcom_pull(task_ids="persist_papers", key="persist_results") if ti else None,
    )
    hybrid_stats = ti.xcom_pull(task_ids="index_papers_hybrid", key="hybrid_index_stats") or {} if ti else {}
    report = build_daily_report(
        download_stats,
        persist_stats,
        hybrid_stats,
        execution_date=context.get("execution_date"),
    )
    if ti:
        ti.xcom_push(key="daily_report", value=report)
    return report

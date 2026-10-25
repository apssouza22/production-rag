"""Run the arXiv ingestion pipeline without Airflow.

Mirrors ``airflow/dags/arxiv_paper_ingestion.py`` (setup → download → parse →
index → report → optional PDF cache cleanup).

Usage (from repo root, with ``.env`` configured):

    uv run python -m src.ingest
    uv run python -m src.ingest --date 20251020
    uv run python -m src.ingest --skip-cleanup
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

from src.config import get_settings
from src.domain.arxiv_ingestion import setup_environment
from src.domain.arxiv_ingestion.fetching import download_papers, parse_pdfs
from src.domain.arxiv_ingestion.indexing import index_papers_hybrid
from src.domain.arxiv_ingestion.reporting import generate_daily_report

logger = logging.getLogger(__name__)

DEFAULT_PDF_RETENTION_DAYS = 30


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )


def _default_target_date() -> str:
    """Same as Airflow: ingest papers for the day before the run."""
    return (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")


def _cleanup_old_pdfs(cache_dir: Path, retention_days: int) -> int:
    """Remove PDFs older than ``retention_days`` from the cache directory."""
    if not cache_dir.is_dir():
        logger.info("PDF cache directory does not exist, skipping cleanup: %s", cache_dir)
        return 0

    cutoff = time.time() - retention_days * 24 * 60 * 60
    removed = 0
    for pdf_path in cache_dir.glob("*.pdf"):
        try:
            if pdf_path.stat().st_mtime < cutoff:
                pdf_path.unlink()
                removed += 1
        except OSError as exc:
            logger.warning("Failed to remove %s: %s", pdf_path, exc)

    logger.info("Cleanup complete: removed %s PDF(s) older than %s days from %s", removed, retention_days, cache_dir)
    return removed


def run_pipeline(
    target_date: str,
    *,
    skip_setup: bool = False,
    skip_cleanup: bool = False,
    pdf_retention_days: int = DEFAULT_PDF_RETENTION_DAYS,
) -> dict:
    """Execute ingestion steps and return the daily report dict."""
    if not skip_setup:
        setup_environment()

    download_results = download_papers(target_date)
    parse_results = parse_pdfs(download_results)
    hybrid_stats = index_papers_hybrid(parse_results)
    report = generate_daily_report(
        download_results,
        parse_results,
        hybrid_stats,
        execution_date=datetime.now(),
    )

    if not skip_cleanup:
        settings = get_settings()
        cache_dir = Path(settings.arxiv.pdf_cache_dir)
        report["cleanup"] = {
            "pdfs_removed": _cleanup_old_pdfs(cache_dir, pdf_retention_days),
            "cache_dir": str(cache_dir),
            "retention_days": pdf_retention_days,
        }

    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the arXiv CS.AI ingestion pipeline (Airflow DAG equivalent).",
    )
    parser.add_argument(
        "--date",
        metavar="YYYYMMDD",
        help="Target ingestion date (default: yesterday, matching the Airflow DAG).",
    )
    parser.add_argument(
        "--skip-setup",
        action="store_true",
        help="Skip DB/OpenSearch verification and index setup.",
    )
    parser.add_argument(
        "--skip-cleanup",
        action="store_true",
        help="Skip removal of cached PDFs older than the retention window.",
    )
    parser.add_argument(
        "--pdf-retention-days",
        type=int,
        default=DEFAULT_PDF_RETENTION_DAYS,
        help=f"Delete cached PDFs older than this many days (default: {DEFAULT_PDF_RETENTION_DAYS}).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the daily report as JSON on stdout.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    _configure_logging(args.verbose)

    target_date = args.date or _default_target_date()
    logger.info("Starting ingestion for target date %s", target_date)

    try:
        report = run_pipeline(
            target_date,
            skip_setup=args.skip_setup,
            skip_cleanup=args.skip_cleanup,
            pdf_retention_days=args.pdf_retention_days,
        )
    except Exception:
        logger.exception("Ingestion pipeline failed")
        return 1

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        logger.info("Ingestion finished successfully")
        logger.info("Report summary: %s", json.dumps(report, default=str))

    return 0


if __name__ == "__main__":
    sys.exit(main())

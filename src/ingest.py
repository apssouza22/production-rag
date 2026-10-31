"""Run the arXiv ingestion pipeline without Airflow.

Mirrors ``airflow/dags/arxiv_paper_ingestion.py`` (setup → download → parse →
index → report → optional PDF cache cleanup).

Usage (from repo root, with ``.env`` configured):

    uv run python -m src.ingest
    uv run python -m src.ingest --date 20251020
    uv run python -m src.ingest --skip-cleanup
    uv run python -m src.ingest --skip-download --pdfs-dir evals/experiments/arxiv_pdfs
    uv run python -m src.ingest --ingestion-name evals-local-pdfs
"""

import argparse
import json
import logging
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

from src.config import get_settings
from src.domain.arxiv_ingestion import setup_environment
from src.domain.arxiv_ingestion.reporting import generate_daily_report
from src.domain.arxiv_ingestion.service import get_ingestion_service

logger = logging.getLogger(__name__)

DEFAULT_PDF_RETENTION_DAYS = 30
INGESTION_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")


def _validate_ingestion_name(name: str) -> str:
    normalized = name.strip().lower()
    if not INGESTION_NAME_PATTERN.fullmatch(normalized):
        raise ValueError(
            "ingestion name must be 1-128 chars, lowercase alphanumeric, and may contain '.', '_', or '-'"
        )
    return normalized


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
    skip_download: bool = False,
    pdfs_dir: Path | None = None,
    pdf_retention_days: int = DEFAULT_PDF_RETENTION_DAYS,
    ingestion_name: str = "dev",
) -> dict:
    """Execute ingestion steps and return the daily report dict."""
    if not skip_setup:
        setup_environment()

    ingestion = get_ingestion_service()
    if skip_download:
        if pdfs_dir is None:
            raise ValueError("pdfs_dir is required when skip_download is True")
        download_results = ingestion.load_papers_from_pdf_directory(pdfs_dir, target_date)
    else:
        download_results = ingestion.download_papers(target_date)
    markdown_results = ingestion.parse_pdfs_to_markdown(download_results)
    persist_results = ingestion.persist_papers(download_results, markdown_results)
    hybrid_stats = ingestion.index_papers_hybrid(persist_results, ingestion_name=ingestion_name)
    report = generate_daily_report(
        download_results,
        persist_results,
        hybrid_stats,
        execution_date=datetime.now(),
    )
    report["ingestion_name"] = ingestion_name

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
        "--ingestion-name",
        metavar="NAME",
        help=(
            "OpenSearch partition for indexed chunks (default: OPENSEARCH__INGESTION_NAME or 'production'). "
            "Set the same value in OPENSEARCH__INGESTION_NAME when querying."
        ),
    )
    parser.add_argument(
        "--skip-setup",
        action="store_true",
        help="Skip DB/OpenSearch verification and index setup.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip arXiv fetch/download; use PDFs from --pdfs-dir (metadata still fetched from arXiv API).",
    )
    parser.add_argument(
        "--pdfs-dir",
        type=Path,
        metavar="DIR",
        help="Directory of *.pdf files to parse when --skip-download is set (required with --skip-download).",
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
    parser = _build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)

    if args.skip_download and args.pdfs_dir is None:
        parser.error("--pdfs-dir is required when using --skip-download")
    if args.pdfs_dir is not None and not args.skip_download:
        parser.error("--pdfs-dir can only be used together with --skip-download")

    pdfs_dir = args.pdfs_dir.resolve() if args.pdfs_dir else None
    if pdfs_dir is not None and not pdfs_dir.is_dir():
        parser.error(f"PDF directory does not exist: {pdfs_dir}")

    target_date = args.date or _default_target_date()
    settings = get_settings()
    raw_ingestion_name = args.ingestion_name or settings.opensearch.ingestion_name
    try:
        ingestion_name = _validate_ingestion_name(raw_ingestion_name)
    except ValueError as exc:
        parser.error(str(exc))

    if args.skip_download:
        logger.info(
            "Starting ingestion for target date %s using local PDFs from %s (ingestion_name=%s)",
            target_date,
            pdfs_dir,
            ingestion_name,
        )
    else:
        logger.info("Starting ingestion for target date %s (ingestion_name=%s)", target_date, ingestion_name)

    try:
        report = run_pipeline(
            target_date,
            skip_setup=args.skip_setup,
            skip_cleanup=args.skip_cleanup,
            skip_download=args.skip_download,
            pdfs_dir=pdfs_dir,
            pdf_retention_days=args.pdf_retention_days,
            ingestion_name=ingestion_name,
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

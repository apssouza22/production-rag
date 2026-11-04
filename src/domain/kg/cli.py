"""CLI entrypoint for backfill KG indexing on explicit arxiv ID lists."""

import argparse
import asyncio
import logging

from src.config import ConfigurationError, get_settings
from src.domain.kg.factory import make_kg_indexing_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _parse_arxiv_ids(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


async def _run_index(args: argparse.Namespace) -> int:
    settings = get_settings()
    if not settings.kg.enabled:
        raise ConfigurationError("KG__ENABLED must be true to run KG indexing")

    arxiv_ids = _parse_arxiv_ids(args.arxiv_ids)
    if not arxiv_ids:
        logger.error("No arxiv IDs provided")
        return 1

    service = make_kg_indexing_service()
    stats = await service.index_arxiv_ids(arxiv_ids, replace=args.replace)
    logger.info("KG indexing complete: %s", stats)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Knowledge graph indexing CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    index_parser = sub.add_parser("index", help="Extract KG from OpenSearch chunks for given arxiv IDs")
    index_parser.add_argument(
        "--arxiv-ids",
        required=True,
        help="Comma-separated arXiv IDs (e.g. 2611.02200,2611.02208)",
    )
    index_parser.add_argument(
        "--replace",
        action="store_true",
        help="Delete existing KG data for each paper before re-extracting",
    )

    args = parser.parse_args()
    try:
        if args.command == "index":
            raise SystemExit(asyncio.run(_run_index(args)))
    except ConfigurationError as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

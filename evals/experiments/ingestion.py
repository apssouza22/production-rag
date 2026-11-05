"""Ingest the experiment PDFs, skipping the run when the partition is already indexed."""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from src.config import Settings, get_settings
from src.domain.arxiv_ingestion import setup_environment
from src.domain.arxiv_ingestion.schemas import DownloadResults
from src.domain.arxiv_ingestion.service import ArxivIngestionService, get_ingestion_service
from src.domain.kg.factory import make_kg_indexing_service
from src.domain.rrf import PaperRrfSearch

from evals.experiments.config import ExperimentSettings
from evals.experiments.schemas import ExperimentIngestionResult, KgIngestionSummary

logger = logging.getLogger(__name__)


def count_indexed_chunks(paper_rrf_search: PaperRrfSearch, ingestion_name: str) -> int:
    """Return how many chunks are indexed for an ingestion partition."""
    if not paper_rrf_search.client.indices.exists(index=paper_rrf_search.index_name):
        return 0

    response = paper_rrf_search.client.count(
        index=paper_rrf_search.index_name,
        body={"query": {"term": {"ingestion_name": ingestion_name}}},
    )
    return int(response["count"])


@dataclass(frozen=True)
class ExperimentPapers:
    """Local PDF metadata resolved through the arXiv ingestion domain."""

    service: ArxivIngestionService
    download_results: DownloadResults

    @property
    def arxiv_ids(self) -> list[str]:
        return [paper.arxiv_id for paper in self.download_results.papers]


@dataclass(frozen=True)
class HybridIngestionOutcome:
    chunk_count: int
    skipped: bool
    papers_stored: int | None = None
    pdfs_parsed: int | None = None
    chunks_indexed: int | None = None


class ExperimentPaperLoader:
    """Load arXiv metadata for PDFs under the experiment data directory."""

    @staticmethod
    def load(pdfs_dir: Path) -> ExperimentPapers:
        setup_environment()
        service = get_ingestion_service()
        target_date = datetime.now().strftime("%Y%m%d")
        download_results = service.load_papers_from_pdf_directory(pdfs_dir, target_date)
        if not download_results.papers:
            raise RuntimeError(f"No arXiv metadata could be resolved for the PDFs in {pdfs_dir}")
        return ExperimentPapers(service=service, download_results=download_results)


class HybridChunkIngestor:
    """Ensure hybrid OpenSearch chunks exist for an experiment ingestion partition."""

    def __init__(self, paper_rrf_search: PaperRrfSearch, ingestion_name: str) -> None:
        self._paper_rrf_search = paper_rrf_search
        self._ingestion_name = ingestion_name

    def partition_chunk_count(self) -> int:
        return count_indexed_chunks(self._paper_rrf_search, self._ingestion_name)

    def ensure(
        self,
        papers: ExperimentPapers,
        *,
        force: bool,
    ) -> HybridIngestionOutcome:
        """Index hybrid chunks when the partition is empty or ``force`` is set."""
        chunk_count = self.partition_chunk_count()
        if chunk_count and not force:
            logger.info(
                "Skipping hybrid ingestion: %s chunks already indexed for ingestion_name=%s",
                chunk_count,
                self._ingestion_name,
            )
            return HybridIngestionOutcome(chunk_count=chunk_count, skipped=True)

        logger.info(
            "Ingesting PDFs into ingestion_name=%s",
            self._ingestion_name,
        )
        markdown_results = papers.service.parse_pdfs_to_markdown(papers.download_results)
        persist_results = papers.service.persist_papers(papers.download_results, markdown_results)
        hybrid_stats = papers.service.index_papers_hybrid(
            persist_results,
            ingestion_name=self._ingestion_name,
            arxiv_ids=papers.arxiv_ids,
        )

        chunk_count = self.partition_chunk_count()
        if not chunk_count:
            created = int(hybrid_stats.get("total_chunks_created", 0))
            indexed = int(hybrid_stats.get("total_chunks_indexed", 0))
            errors = int(hybrid_stats.get("total_errors", 0))
            if created > 0 and indexed == 0:
                raise RuntimeError(
                    f"Chunking created {created} chunk(s) but OpenSearch indexed none for "
                    f"ingestion_name={self._ingestion_name} (indexing errors={errors}). "
                    "Check logs for strict mapping or bulk index failures; ensure hybrid index "
                    "setup ran (restart stack or run setup_indices)."
                )
            raise RuntimeError(
                f"Ingestion produced no chunks for ingestion_name={self._ingestion_name} "
                f"(chunks_created={created}, indexing_errors={errors}). "
                "Ensure PDFs parsed successfully and papers have raw_text in Postgres."
            )

        logger.info(
            "Hybrid ingestion complete: %s chunks indexed for ingestion_name=%s",
            chunk_count,
            self._ingestion_name,
        )
        return HybridIngestionOutcome(
            chunk_count=chunk_count,
            skipped=False,
            papers_stored=persist_results.papers_stored,
            pdfs_parsed=int(markdown_results.get("pdfs_parsed", 0)),
            chunks_indexed=int(hybrid_stats.get("total_chunks_indexed", 0)),
        )


class KgGraphIngestor:
    """Backfill Postgres graph and KG vector indices from indexed chunks when KG is enabled."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    def ensure(self, arxiv_ids: list[str], *, force: bool) -> KgIngestionSummary | None:
        if not self._settings.kg.enabled:
            return None

        logger.info("Running KG indexing for %s paper(s), replace=%s", len(arxiv_ids), force)
        kg_service = make_kg_indexing_service(settings=self._settings)
        stats = asyncio.run(kg_service.index_arxiv_ids(arxiv_ids, replace=force))
        logger.info("KG indexing complete: %s", stats)
        return KgIngestionSummary.model_validate(stats)


class ExperimentIngestor:
    """Orchestrate local-PDF hybrid ingest and optional KG indexing for eval experiments."""

    def __init__(
        self,
        paper_rrf_search: PaperRrfSearch,
        settings: ExperimentSettings,
        *,
        force: bool = False,
    ) -> None:
        self._settings = settings
        self._force = force
        self._pdfs_dir = settings.experiment_pdfs_dir.resolve()
        self._ingestion_name = settings.experiment_ingestion_name
        self._hybrid = HybridChunkIngestor(paper_rrf_search, self._ingestion_name)
        self._kg = KgGraphIngestor()

    def run(self) -> ExperimentIngestionResult:
        if not self._pdfs_dir.is_dir():
            raise FileNotFoundError(f"Experiment PDF directory does not exist: {self._pdfs_dir}")

        papers = ExperimentPaperLoader.load(self._pdfs_dir)
        hybrid = self._hybrid.ensure(papers, force=self._force)
        kg_summary = self._kg.ensure(papers.arxiv_ids, force=self._force)

        return ExperimentIngestionResult(
            skipped=hybrid.skipped,
            chunk_count=hybrid.chunk_count,
            ingestion_name=self._ingestion_name,
            pdfs_dir=str(self._pdfs_dir),
            arxiv_ids=papers.arxiv_ids,
            papers_stored=hybrid.papers_stored,
            pdfs_parsed=hybrid.pdfs_parsed,
            chunks_indexed=hybrid.chunks_indexed,
            kg=kg_summary,
        )


def ensure_ingested(
    paper_rrf_search: PaperRrfSearch,
    settings: ExperimentSettings,
    *,
    force: bool = False,
) -> dict[str, object]:
    """Run hybrid chunk ingest and optional KG indexing for local experiment PDFs."""
    return ExperimentIngestor(paper_rrf_search, settings, force=force).run().model_dump_for_runner()

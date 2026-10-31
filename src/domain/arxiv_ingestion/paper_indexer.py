import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from src.domain.db.interfaces.base import BaseDatabase
from src.domain.indexing.factory import make_hybrid_indexing_service

logger = logging.getLogger(__name__)


class PaperIndexer:
    """Indexes persisted papers for hybrid search (chunking and vector embeddings)."""

    def __init__(self, database: BaseDatabase) -> None:
        self.database = database

    def index_papers_hybrid(
        self,
        persist_results: dict[str, Any] | None = None,
        *,
        ingestion_name: str = "production",
    ) -> dict[str, Any]:
        """Index papers with chunking and vector embeddings for hybrid search."""
        try:
            from src.domain.paper import Paper

            with self.database.get_session() as session:
                if persist_results and persist_results.get("papers_stored", 0) > 0:
                    from sqlalchemy import desc

                    papers = (
                        session.query(Paper)
                        .order_by(desc(Paper.created_at))
                        .limit(persist_results["papers_stored"])
                        .all()
                    )
                else:
                    cutoff_date = datetime.now(timezone.utc) - timedelta(days=1)
                    papers = session.query(Paper).filter(Paper.created_at >= cutoff_date).all()

                if not papers:
                    logger.info("No papers to index for hybrid search")
                    return {"papers_indexed": 0, "chunks_created": 0}

                logger.info("Indexing %s papers for hybrid search", len(papers))
                stats = asyncio.run(self._index_papers_with_chunks(papers, ingestion_name=ingestion_name))
                logger.info(
                    "Hybrid indexing complete: %s papers, %s chunks created, %s chunks indexed",
                    stats["papers_processed"],
                    stats["total_chunks_created"],
                    stats["total_chunks_indexed"],
                )
                return stats
        except Exception as e:
            logger.error("Failed to index papers for hybrid search: %s", e)
            raise

    async def _index_papers_with_chunks(self, papers: list[Any], ingestion_name: str) -> dict[str, Any]:
        """Index papers with chunking and embeddings."""
        indexing_service = make_hybrid_indexing_service()

        papers_data = []
        for paper in papers:
            if hasattr(paper, "__dict__"):
                paper_dict = {
                    "id": str(paper.id),
                    "arxiv_id": paper.arxiv_id,
                    "title": paper.title,
                    "authors": paper.authors,
                    "abstract": paper.abstract,
                    "categories": paper.categories,
                    "published_date": paper.published_date,
                    "raw_text": paper.raw_text,
                    "sections": paper.sections,
                }
            else:
                paper_dict = paper
            papers_data.append(paper_dict)

        return await indexing_service.index_papers_batch(
            papers=papers_data,
            replace_existing=True,
            ingestion_name=ingestion_name,
        )


def make_paper_indexer(database: BaseDatabase) -> PaperIndexer:
    """Create PaperIndexer with configured database access."""
    return PaperIndexer(database=database)

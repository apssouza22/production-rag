import logging
from datetime import datetime
from typing import Any

from dateutil import parser as date_parser
from sqlalchemy.orm import Session

from src.domain.arxiv.paper import ArxivPaper, PaperCreate
from src.domain.paper import PaperRepository
from src.domain.pdf_parser.models import ParsedPaper

logger = logging.getLogger(__name__)


class PaperPersist:
    """Persists arXiv paper metadata and parsed PDF content to PostgreSQL."""

    def store_papers(self, papers: list[ArxivPaper], parsed_papers: dict[str, ParsedPaper], db_session: Session) -> int:
        """Persist parsed papers and metadata to PostgreSQL."""
        return self._store_papers_to_db(papers, parsed_papers, db_session)

    @staticmethod
    def _serialize_parsed_content(parsed_paper: ParsedPaper) -> dict[str, Any]:
        """Serialize ParsedPaper content for database storage."""
        try:
            pdf_content = parsed_paper.pdf_content

            sections = [
                {
                    "title": section.title,
                    "content": section.content,
                    "level": section.level,
                    "breadcrumb": section.breadcrumb,
                    "figure_paths": section.figure_paths,
                    "table_paths": section.table_paths,
                }
                for section in pdf_content.sections
            ]

            references = list(pdf_content.references)

            return {
                "raw_text": pdf_content.raw_text,
                "sections": sections,
                "references": references,
                "parser_used": pdf_content.parser_used.value if pdf_content.parser_used else None,
                "parser_metadata": pdf_content.metadata or {},
                "pdf_processed": True,
                "pdf_processing_date": datetime.now(),
            }
        except Exception as e:
            logger.error(f"Failed to serialize parsed content: {e}")
            return {"pdf_processed": False, "parser_metadata": {"error": str(e)}}

    def _store_papers_to_db(
        self,
        papers: list[ArxivPaper],
        parsed_papers: dict[str, ParsedPaper],
        db_session: Session,
    ) -> int:
        """Store papers and parsed content to the database."""
        paper_repo = PaperRepository(db_session)
        stored_count = 0

        for paper in papers:
            try:
                parsed_paper = parsed_papers.get(paper.arxiv_id)

                published_date = (
                    date_parser.parse(paper.published_date) if isinstance(paper.published_date, str) else paper.published_date
                )
                paper_data = {
                    "arxiv_id": paper.arxiv_id,
                    "title": paper.title,
                    "authors": paper.authors,
                    "abstract": paper.abstract,
                    "categories": paper.categories,
                    "published_date": published_date,
                    "pdf_url": paper.pdf_url,
                }

                if parsed_paper:
                    parsed_content = self._serialize_parsed_content(parsed_paper)
                    paper_data.update(parsed_content)
                    logger.debug(
                        f"Storing paper {paper.arxiv_id} with parsed content ({len(parsed_content.get('raw_text', '')) if parsed_content.get('raw_text') else 0} chars)"
                    )
                else:
                    paper_data.update(
                        {
                            "pdf_processed": False,
                            "parser_metadata": {"note": "PDF processing not available or failed"}
                        }
                    )
                    logger.debug(f"Storing paper {paper.arxiv_id} with metadata only")

                paper_create = PaperCreate(**paper_data)
                stored_paper = paper_repo.upsert(paper_create)

                if stored_paper:
                    stored_count += 1
                    content_info = "with parsed content" if parsed_paper else "metadata only"
                    logger.debug(f"Stored paper {paper.arxiv_id} to database ({content_info})")

            except Exception as e:
                logger.error(f"Failed to store paper {paper.arxiv_id}: {e}")

        try:
            db_session.commit()
            logger.info(f"Committed {stored_count} papers to database with full content storage")
        except Exception as e:
            logger.error(f"Failed to commit papers to database: {e}")
            db_session.rollback()
            stored_count = 0

        return stored_count


def make_paper_persist() -> PaperPersist:
    """Create PaperPersist instance."""
    return PaperPersist()

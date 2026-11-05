"""Generate evaluation questions from the chunks indexed by the experiment ingestion."""

import json
import logging
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from src.domain.rrf import HybridRrfSearch

from evals.experiments.config import ExperimentSettings
from evals.experiments.schemas import ExperimentQuery, GeneratedQuestions

logger = logging.getLogger(__name__)

MAX_CHUNKS_SCANNED = 2000

QUESTION_SYSTEM_PROMPT = (
    "You write evaluation questions for a retrieval system over research papers. "
    "Using only the provided excerpts from a single paper, write {count} diverse, self-contained "
    "questions a researcher might ask, each with a concise ground-truth answer taken from the excerpts. "
    "Questions must be answerable from the excerpts alone and must not refer to 'the excerpt', "
    "'the text', or 'this paper' — name the concept, method, or result instead."
)


def load_chunks_by_paper(
    paper_rrf_search: HybridRrfSearch,
    ingestion_name: str,
    chunks_per_paper: int,
) -> dict[str, list[str]]:
    """Return up to ``chunks_per_paper`` chunk texts per paper in the ingestion partition."""
    response = paper_rrf_search.client.search(
        index=paper_rrf_search.index_name,
        body={
            "query": {"term": {"ingestion_name": ingestion_name}},
            "size": MAX_CHUNKS_SCANNED,
            "sort": [{"document_id": "asc"}, {"chunk_index": "asc"}],
            "_source": ["document_id", "chunk_text", "metadata.arxiv_id", "metadata.title"],
        },
    )

    chunks_by_paper: dict[str, list[str]] = {}
    for hit in response["hits"]["hits"]:
        source = hit["_source"]
        chunk_text = (source.get("chunk_text") or "").strip()
        if not chunk_text:
            continue
        paper_key = source.get("metadata", {}).get("arxiv_id") or source.get("document_id", "")
        if not paper_key:
            continue
        chunks = chunks_by_paper.setdefault(paper_key, [])
        if len(chunks) < chunks_per_paper:
            chunks.append(chunk_text)

    logger.info("Loaded chunks for %s paper(s) from ingestion_name=%s", len(chunks_by_paper), ingestion_name)
    return chunks_by_paper


def load_cached_queries(queries_path: Path) -> list[ExperimentQuery]:
    """Load previously generated queries from disk."""
    with queries_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, list):
        raise ValueError(f"Query cache must contain a JSON list: {queries_path}")

    return [ExperimentQuery.model_validate(item) for item in payload]


def write_queries(queries: list[ExperimentQuery], queries_path: Path) -> None:
    """Persist generated queries so later runs reuse the same dataset."""
    queries_path.parent.mkdir(parents=True, exist_ok=True)
    with queries_path.open("w", encoding="utf-8") as handle:
        json.dump([query.model_dump() for query in queries], handle, indent=2)


class QuestionGenerator:
    """Build a question/reference dataset from the indexed paper chunks."""

    def __init__(self, paper_rrf_search: HybridRrfSearch, chat_model: BaseChatModel, settings: ExperimentSettings) -> None:
        self.paper_rrf_search = paper_rrf_search
        self.chat_model = chat_model
        self.settings = settings

    async def build_queries(self, *, regenerate: bool = False) -> list[ExperimentQuery]:
        """Return cached queries when available, otherwise generate and cache them."""
        queries_path = self.settings.experiment_queries_path
        if queries_path.is_file() and not regenerate:
            queries = load_cached_queries(queries_path)
            logger.info("Loaded %s cached queries from %s", len(queries), queries_path)
            return queries

        chunks_by_paper = load_chunks_by_paper(
            self.paper_rrf_search,
            self.settings.experiment_ingestion_name,
            self.settings.experiment_chunks_per_paper,
        )
        if not chunks_by_paper:
            raise RuntimeError(
                f"No chunks found for ingestion_name={self.settings.experiment_ingestion_name}; cannot generate questions"
            )

        queries: list[ExperimentQuery] = []
        for arxiv_id, chunks in chunks_by_paper.items():
            queries.extend(await self._generate_for_paper(arxiv_id, chunks))

        if not queries:
            raise RuntimeError("Question generation produced no queries")

        write_queries(queries, queries_path)
        logger.info("Generated %s queries and cached them at %s", len(queries), queries_path)
        return queries

    async def _generate_for_paper(self, arxiv_id: str, chunks: list[str]) -> list[ExperimentQuery]:
        """Ask the eval model for question/reference pairs grounded in one paper."""
        count = self.settings.experiment_questions_per_paper
        excerpts = "\n\n".join(f"[{index + 1}] {chunk}" for index, chunk in enumerate(chunks))
        generator = self.chat_model.with_structured_output(GeneratedQuestions)

        response = await generator.ainvoke(
            [
                SystemMessage(content=QUESTION_SYSTEM_PROMPT.format(count=count)),
                HumanMessage(content=f"Paper: {arxiv_id}\n\nExcerpts:\n{excerpts}"),
            ]
        )
        parsed = response if isinstance(response, GeneratedQuestions) else GeneratedQuestions.model_validate(response)

        queries = [
            ExperimentQuery(user_input=item.question.strip(), reference=item.reference.strip(), arxiv_id=arxiv_id)
            for item in parsed.questions[:count]
            if item.question.strip() and item.reference.strip()
        ]
        logger.info("Generated %s question(s) for %s", len(queries), arxiv_id)
        return queries

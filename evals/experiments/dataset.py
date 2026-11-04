"""Build evaluation samples from rerank search retrieval over the experiment partition."""

import logging

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from src.domain.search.service import SearchService

from evals.core.schemas import EvalSample
from evals.experiments.config import ExperimentSettings
from evals.experiments.schemas import ExperimentQuery

logger = logging.getLogger(__name__)

ANSWER_SYSTEM_PROMPT = (
    "You answer questions using only the provided research paper excerpts. "
    "If the excerpts do not contain enough information, say you do not know."
)


class ExperimentDataset:
    """Retrieve contexts per query and generate an answer grounded in them."""

    source_name = "experiment_local_pdfs"

    def __init__(
        self,
        queries: list[ExperimentQuery],
        rerank_search_service: SearchService,
        chat_model: BaseChatModel,
        settings: ExperimentSettings,
    ) -> None:
        self.queries = queries
        self.rerank_search_service = rerank_search_service
        self.chat_model = chat_model
        self.settings = settings
        self.total_items = len(queries)
        self.source_location = str(settings.experiment_queries_path)

    async def load(self) -> list[EvalSample]:
        samples: list[EvalSample] = []
        for query in self.queries:
            sample = await self._build_sample(query)
            if sample is not None:
                samples.append(sample)
        return samples

    async def _build_sample(self, query: ExperimentQuery) -> EvalSample | None:
        result = await self.rerank_search_service.search_rrf(query.user_input, top_k=self.settings.experiment_top_k)
        contexts = [document.chunk_text for document in result.after_rerank if document.chunk_text.strip()]
        if not contexts:
            logger.warning("Skipping query with no retrieved contexts: %s", query.user_input)
            return None

        return EvalSample(
            input=query.user_input,
            output=await self._generate_response(query.user_input, contexts),
            reference=query.reference,
            contexts=contexts,
            source_id=query.arxiv_id or None,
            metadata={
                "rerank_applied": result.rerank_applied,
                "search_mode": result.search_mode,
                "retrieved_arxiv_ids": sorted({document.arxiv_id for document in result.after_rerank}),
            },
        )

    async def _generate_response(self, user_input: str, contexts: list[str]) -> str:
        context_block = "\n\n".join(f"[{index + 1}] {context}" for index, context in enumerate(contexts))
        response = await self.chat_model.ainvoke(
            [
                SystemMessage(content=ANSWER_SYSTEM_PROMPT),
                HumanMessage(content=f"Question: {user_input}\n\nExcerpts:\n{context_block}"),
            ]
        )
        content = response.text
        if not content:
            raise RuntimeError(f"Empty response generated for query: {user_input}")
        return content.strip()

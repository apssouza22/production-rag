"""Build Ragas evaluation samples from queries and rerank search retrieval."""

import json
import logging
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from evals.core.schemas import EvalSample
from evals.rag_evals.config import RagEvalSettings
from evals.rag_evals.retrieval import RerankSearchClient
from evals.rag_evals.schemas import EvalQuery

logger = logging.getLogger(__name__)

ANSWER_SYSTEM_PROMPT = (
    "You answer questions using only the provided research paper excerpts. "
    "If the excerpts do not contain enough information, say you do not know."
)


def load_queries(dataset_path: Path) -> list[EvalQuery]:
    """Load evaluation queries from a JSON file."""
    with dataset_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, list):
        raise ValueError("Dataset file must contain a JSON list of query objects")

    return [EvalQuery.model_validate(item) for item in payload]


class DatasetBuilder:
    """Populate retrieved contexts and responses for Ragas evaluation."""

    def __init__(
        self,
        settings: RagEvalSettings,
        retrieval_client: RerankSearchClient,
        chat_model: BaseChatModel | None = None,
    ) -> None:
        self.settings = settings
        self.retrieval_client = retrieval_client
        self.chat_model = chat_model

    async def build_sample_from_inputs(
        self,
        user_input: str,
        reference: str = "",
        response: str | None = None,
    ) -> EvalSample | None:
        """Retrieve contexts for one query and prepare a Ragas sample."""
        search_result = await self.retrieval_client.search(user_input)
        retrieved_contexts = RerankSearchClient.contexts_from_result(search_result)
        if not retrieved_contexts:
            logger.warning("Skipping query with no retrieved contexts: %s", user_input)
            return None

        resolved_response = response
        if not resolved_response:
            if not self.settings.rag_eval_generate_responses:
                logger.warning(
                    "Skipping query without response and generation disabled: %s",
                    user_input,
                )
                return None
            resolved_response = await self._generate_response(user_input, retrieved_contexts)

        return EvalSample(
            input=user_input,
            reference=reference,
            output=resolved_response,
            contexts=retrieved_contexts,
            metadata={"rerank_applied": search_result.rerank_applied},
        )

    async def build_samples(self, queries: list[EvalQuery]) -> list[EvalSample]:
        """Retrieve contexts for each query and prepare Ragas samples."""
        samples: list[EvalSample] = []
        for query in queries:
            sample = await self.build_sample_from_inputs(
                user_input=query.user_input,
                reference=query.reference,
                response=query.response,
            )
            if sample is not None:
                samples.append(sample)
        return samples

    async def _generate_response(self, user_input: str, retrieved_contexts: list[str]) -> str:
        """Generate a grounded answer from retrieved contexts for response-based metrics."""
        if self.chat_model is None:
            raise RuntimeError("Chat model is required to generate responses")

        context_block = "\n\n".join(f"[{index + 1}] {context}" for index, context in enumerate(retrieved_contexts))
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


class RagDataset:
    """Load a local query file and enrich it into normalized RAG samples."""

    source_name = "json_rag"

    def __init__(self, dataset_path: Path, builder: DatasetBuilder) -> None:
        self.dataset_path = dataset_path
        self.builder = builder
        self.total_items = 0
        self.total_queries = 0

    async def load(self) -> list[EvalSample]:
        queries = load_queries(self.dataset_path)
        self.total_items = len(queries)
        self.total_queries = len(queries)
        return await self.builder.build_samples(queries)

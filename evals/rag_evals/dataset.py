"""Build Ragas evaluation samples from queries and rerank search retrieval."""

import json
import logging
from pathlib import Path

from openai import AsyncOpenAI

from rag_evals.config import RagEvalSettings
from rag_evals.retrieval import RerankSearchClient
from rag_evals.schemas import EvalQuery, EvalSample

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
        openai_client: AsyncOpenAI | None = None,
    ) -> None:
        self.settings = settings
        self.retrieval_client = retrieval_client
        self.openai_client = openai_client

    async def build_samples(self, queries: list[EvalQuery]) -> list[EvalSample]:
        """Retrieve contexts for each query and prepare Ragas samples."""
        samples: list[EvalSample] = []
        for query in queries:
            search_result = await self.retrieval_client.search(query.user_input)
            retrieved_contexts = RerankSearchClient.contexts_from_result(search_result)
            if not retrieved_contexts:
                logger.warning("Skipping query with no retrieved contexts: %s", query.user_input)
                continue

            response = query.response
            if not response:
                if not self.settings.rag_eval_generate_responses:
                    logger.warning(
                        "Skipping query without response and generation disabled: %s",
                        query.user_input,
                    )
                    continue
                response = await self._generate_response(query.user_input, retrieved_contexts)

            samples.append(
                EvalSample(
                    user_input=query.user_input,
                    response=response,
                    retrieved_contexts=retrieved_contexts,
                    rerank_applied=search_result.rerank_applied,
                )
            )
        return samples

    async def _generate_response(self, user_input: str, retrieved_contexts: list[str]) -> str:
        """Generate a grounded answer from retrieved contexts for response-based metrics."""
        if self.openai_client is None:
            raise RuntimeError("OpenAI client is required to generate responses")

        context_block = "\n\n".join(f"[{index + 1}] {context}" for index, context in enumerate(retrieved_contexts))
        completion = await self.openai_client.chat.completions.create(
            model=self.settings.evaluation_llm,
            messages=[
                {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Question: {user_input}\n\nExcerpts:\n{context_block}",
                },
            ],
        )
        content = completion.choices[0].message.content
        if not content:
            raise RuntimeError(f"Empty response generated for query: {user_input}")
        return content.strip()

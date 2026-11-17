"""Knowledge-source routing via OpenAI Decisions, with structured-output fallback."""

import logging

from langchain_core.language_models import BaseChatModel
from langchain_openai.decisions import Predicate

from src.config import Settings
from src.platform.decisions import DecisionsClient

from .prompts import (
    CLASSIFY_SYSTEM_PROMPT,
    ROUTER_DATABASE_INSTRUCTIONS,
    ROUTER_DOCUMENTS_INSTRUCTIONS,
    ROUTER_SOURCE_THRESHOLD,
)
from .schemas import ClassificationResult

logger = logging.getLogger(__name__)

_KNOWLEDGE_SOURCES = ("documents", "database")


class KnowledgeSourceClassifier:
    """Route a query to ``documents`` and/or ``database`` knowledge sources."""

    def __init__(
        self,
        query: str,
        settings: Settings,
        router_model_name: str,
        *,
        router_chat_model: BaseChatModel,
    ) -> None:
        self._query = query
        self._settings = settings
        self._router_model_name = router_model_name
        self._router_chat_model = router_chat_model
        self._decisions = DecisionsClient(settings, router_model_name)

    async def classify(self) -> list[dict[str, str]]:
        try:
            return await self._classify_with_decisions()
        except Exception as exc:
            logger.warning(
                "Knowledge router decisions call failed (%s), falling back to structured output",
                exc,
            )
            return await self._classify_with_structured_output()

    async def _classify_with_decisions(self) -> list[dict[str, str]]:
        response = await self._decisions.invoke(
            self._query,
            {
                "documents": Predicate(instructions=ROUTER_DOCUMENTS_INSTRUCTIONS),
                "database": Predicate(instructions=ROUTER_DATABASE_INSTRUCTIONS),
            },
        )
        classifications: list[dict[str, str]] = []
        for source in _KNOWLEDGE_SOURCES:
            if source in DecisionsClient.refusal_names(response):
                continue
            if source not in response.predicates:
                continue
            if response.predicates[source].probability >= ROUTER_SOURCE_THRESHOLD:
                classifications.append({"source": source, "query": self._query})
        if not classifications:
            raise ValueError("No knowledge sources exceeded the decisions threshold")
        return classifications

    async def _classify_with_structured_output(self) -> list[dict[str, str]]:
        structured_llm = self._router_chat_model.with_structured_output(ClassificationResult)
        result = await structured_llm.ainvoke(
            [
                {"role": "system", "content": CLASSIFY_SYSTEM_PROMPT},
                {"role": "user", "content": self._query},
            ]
        )
        return [{"source": item.source, "query": item.query} for item in result.classifications]


async def classify_knowledge_sources(
    query: str,
    settings: Settings,
    router_model_name: str,
    *,
    router_chat_model: BaseChatModel,
) -> list[dict[str, str]]:
    """Choose knowledge sources via Decisions predicates, falling back to structured chat output."""
    return await KnowledgeSourceClassifier(
        query,
        settings,
        router_model_name,
        router_chat_model=router_chat_model,
    ).classify()

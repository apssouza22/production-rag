import asyncio

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage

from src.domain.kg.prompts import ENTITY_RELATION_EXTRACTION_PROMPT, KEYWORD_EXTRACTION_PROMPT
from src.domain.kg.schemas import ChunkExtractionResult, DualKeywords


class KgExtractionService:
    """LLM calls for (1) entity/relation extraction at index time and (2) dual keywords at query time."""

    def __init__(
        self,
        chat_model: BaseChatModel,
        *,
        max_entities_per_chunk: int,
        max_relations_per_chunk: int,
        max_concurrent: int,
    ):
        self.chat_model = chat_model
        self.max_entities = max_entities_per_chunk
        self.max_relations = max_relations_per_chunk
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def extract_from_chunk(
        self,
        chunk_text: str,
        *,
        section_title: str = "",
    ) -> ChunkExtractionResult:
        prompt = ENTITY_RELATION_EXTRACTION_PROMPT.format(
            chunk_text=chunk_text,
            section_title=section_title or "N/A",
            max_entities=self.max_entities,
            max_relations=self.max_relations,
        )
        structured = self.chat_model.with_structured_output(ChunkExtractionResult)
        async with self._semaphore:
            result = await structured.ainvoke([HumanMessage(content=prompt)])
        if not isinstance(result, ChunkExtractionResult):
            return ChunkExtractionResult()
        return ChunkExtractionResult(
            entities=result.entities[: self.max_entities],
            relations=result.relations[: self.max_relations],
        )

    async def extract_keywords(self, query: str) -> DualKeywords:
        prompt = KEYWORD_EXTRACTION_PROMPT.format(query=query)
        structured = self.chat_model.with_structured_output(DualKeywords)
        result = await structured.ainvoke([HumanMessage(content=prompt)])
        if not isinstance(result, DualKeywords):
            return DualKeywords()
        return result

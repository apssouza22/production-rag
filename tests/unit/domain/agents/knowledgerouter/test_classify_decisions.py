from unittest.mock import AsyncMock, Mock, patch

import pytest
from langchain_openai.decisions import DecisionResponse, PredicateAnswer
from src.agents.knowledgerouter.classify_decisions import classify_knowledge_sources
from src.config import Settings


@pytest.mark.asyncio
async def test_classify_knowledge_sources_both_predicates():
    settings = Settings(openai_api_key="k", agent_model="gpt-6-luna")
    response = DecisionResponse(
        model="gpt-test",
        answers={
            "documents": PredicateAnswer(type="predicate", probability=0.9),
            "database": PredicateAnswer(type="predicate", probability=0.8),
        },
    )
    mock_decisions = Mock()
    mock_decisions.ainvoke = AsyncMock(return_value=response)
    router_chat = Mock()

    with patch(
        "src.platform.decisions.client.make_openai_decisions",
        return_value=mock_decisions,
    ):
        result = await classify_knowledge_sources(
            "How many BERT papers and what do they propose?",
            settings,
            "gpt-6-luna",
            router_chat_model=router_chat,
        )

    assert len(result) == 2
    assert {row["source"] for row in result} == {"documents", "database"}
    assert all(row["query"].startswith("How many BERT") for row in result)
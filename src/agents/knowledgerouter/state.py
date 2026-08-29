import operator
from typing import Annotated, Literal, TypedDict

KnowledgeSource = Literal["documents", "database"]


class AgentInput(TypedDict):
    """Simple input state passed to each sub-agent."""

    query: str


class AgentOutput(TypedDict):
    """Output from each sub-agent."""

    source: KnowledgeSource
    result: str
    metadata: dict


class Classification(TypedDict):
    """A single routing decision: which agent to call with what query."""

    source: KnowledgeSource
    query: str


class RouterState(TypedDict):
    """Main workflow state for the knowledge router."""

    query: str
    classifications: list[Classification]
    results: Annotated[list[AgentOutput], operator.add]
    final_answer: str

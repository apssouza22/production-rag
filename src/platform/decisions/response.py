"""Provider-agnostic decision outcomes used by :class:`~src.platform.decisions.client.DecisionsClient`."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class NormalizedChoiceAnswer:
    """A categorical decision with the selected label and model confidence."""

    choice: str
    confidence: float


@dataclass(frozen=True)
class NormalizedDecisionResponse:
    """Typed answers from any supported decision backend in one shape.

    Backends map provider-specific payloads (OpenAI ``predicates``, TypeSafe
    ``nouls``, refusals, and so on) into this structure so agent code never
    branches on the active provider.
    """

    model: str
    nouls: dict[str, float] = field(default_factory=dict)
    choices: dict[str, NormalizedChoiceAnswer] = field(default_factory=dict)
    refusals: frozenset[str] = frozenset()

    def refusal_names(self) -> set[str]:
        """Return question IDs the backend refused to answer."""
        return set(self.refusals)

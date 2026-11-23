"""TypeSafe (Jev) decision backend using :mod:`langchain_typesafe`."""

from langchain_core.decisions import Question, State

from src.config import Settings

from ..adapters.responses import from_classifier_response
from ..exceptions import DecisionsProviderError
from ..response import NormalizedDecisionResponse

_DEFAULT_BASE_URL = "https://api.typesafe.ai"
_INSTALL_HINT = "Install the optional dependency: uv sync --group typesafe"


def make_typesafe_classifier(
    settings: Settings,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
):
    """Construct a :class:`langchain_typesafe.TypeSafeClassifier` from settings."""
    try:
        from langchain_typesafe import TypeSafeClassifier
    except ImportError as exc:  # pragma: no cover - exercised via factory tests
        raise DecisionsProviderError("typesafe", f"langchain-typesafe is not installed. {_INSTALL_HINT}") from exc

    resolved_key = api_key or settings.typesafe_api_key
    resolved_base = (base_url or settings.typesafe_base_url or _DEFAULT_BASE_URL).rstrip("/")
    return TypeSafeClassifier(
        api_key=resolved_key or None,
        base_url=resolved_base,
        timeout=float(settings.llm_request_timeout),
    )


class TypeSafeDecisionBackend:
    """Invoke TypeSafe/Jev classifiers with portable decision questions.

    TypeSafe follows the shared :mod:`langchain_core.decisions` wire format, so
    questions and the ``state`` invoke key require no translation—only response
    normalization.
    """

    def __init__(self, classifier) -> None:
        self._classifier = classifier

    async def ainvoke(
        self,
        state: State,
        questions: dict[str, Question],
    ) -> NormalizedDecisionResponse:
        """Run the TypeSafe classifier and return a normalized response."""
        raw = await self._classifier.ainvoke({"state": state, "questions": questions})
        return from_classifier_response(raw)

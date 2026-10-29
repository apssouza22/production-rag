"""Dataset source contract for evaluations."""

from typing import Protocol

from evals.core.schemas import EvalSample


class Dataset(Protocol):
    """Load normalized samples from an evaluation source."""

    async def load(self) -> list[EvalSample]:
        """Return all samples available for this evaluation run."""
        ...

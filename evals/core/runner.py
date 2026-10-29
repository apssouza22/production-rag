"""Runner contract for configuring and executing evaluations."""

from abc import ABC, abstractmethod
from typing import Any


class EvalRunner(ABC):
    """Configure an evaluation source and turn its response into output."""

    @abstractmethod
    async def run_eval(self, generate_report_file: bool = True) -> dict[str, Any]:
        """Execute an evaluation and return its serialized report."""

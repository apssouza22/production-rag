"""Langfuse trace dataset adapter."""

import logging
from datetime import datetime, timedelta

from langfuse import Langfuse

from evals.core.schemas import EvalSample
from evals.live_req_evals.helpers import get_input_output

logger = logging.getLogger(__name__)


class LangfuseTraceDataset:
    """Load unscored Langfuse traces as normalized evaluation samples."""

    source_name = "langfuse_traces"

    def __init__(
        self,
        langfuse: Langfuse,
        *,
        lookback_hours: int = 24,
        limit: int = 100,
        source_location: str | None = None,
    ) -> None:
        self.langfuse = langfuse
        self.lookback_hours = lookback_hours
        self.limit = limit
        self.source_location = source_location
        self.total_items = 0

    async def load(self) -> list[EvalSample]:
        from_timestamp = datetime.now() - timedelta(hours=self.lookback_hours)
        logger.info("Fetching Langfuse traces from %s", from_timestamp)
        try:
            traces = self.langfuse.api.trace.list(
                from_timestamp=from_timestamp,
                order_by="timestamp.asc",
                limit=self.limit,
            ).data
        except Exception:
            logger.exception("Error fetching traces")
            return []

        samples: list[EvalSample] = []
        for trace in traces:
            if trace.scores:
                continue
            input_text, output_text = get_input_output(trace)
            samples.append(
                EvalSample(
                    input=input_text,
                    output=output_text,
                    source_id=trace.id,
                    metadata={"source": "langfuse_trace"},
                )
            )
        self.total_items = len(samples)
        return samples

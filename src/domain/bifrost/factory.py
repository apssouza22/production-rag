from src.config import Settings, get_settings
from src.domain.bifrost.client import BifrostClient


def make_bifrost_client(settings: Settings | None = None) -> BifrostClient:
    """Create and return a Bifrost client instance."""
    resolved_settings = settings or get_settings()
    return BifrostClient(resolved_settings)

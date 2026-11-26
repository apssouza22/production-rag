"""Errors raised when configuring or invoking decision backends."""


class DecisionsProviderError(RuntimeError):
    """The selected decisions provider cannot be constructed.

    Typical causes include a missing optional integration package or absent API
    credentials in application settings.
    """

    def __init__(self, provider: str, message: str) -> None:
        self.provider = provider
        super().__init__(f"[{provider}] {message}")

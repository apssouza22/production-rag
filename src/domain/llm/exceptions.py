class LLMException(Exception):
    """Base exception for LLM-related errors."""


class LLMConnectionError(LLMException):
    """Exception raised when an LLM backend cannot be reached."""


class LLMTimeoutError(LLMException):
    """Exception raised when an LLM backend times out."""

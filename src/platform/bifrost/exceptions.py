class BifrostException(Exception):
    """Exception raised for Bifrost gateway errors."""


class BifrostConnectionError(BifrostException):
    """Exception raised when Bifrost gateway cannot be reached."""


class BifrostTimeoutError(BifrostException):
    """Exception raised when Bifrost gateway times out."""

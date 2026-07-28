class ParsingException(Exception):
    """Base exception for parsing-related errors."""


class PDFParsingException(ParsingException):
    """Base exception for PDF parsing-related errors."""


class PDFValidationError(PDFParsingException):
    """Exception raised when PDF file validation fails."""

"""Abstract base class for all parser implementations."""

from abc import ABC, abstractmethod
from parser_benchmark.models import ParseResult


class BaseParser(ABC):
    """Abstract base class that all parsers must inherit from.

    Defines the interface for parsing tool calls from text.

    Example:
        class MyParser(BaseParser):
            @property
            def name(self) -> str:
                return "my-parser"

            def parse(self, text: str) -> ParseResult:
                # Implementation here
                pass
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this parser.

        Used in benchmarks and logging.
        """
        pass

    @abstractmethod
    def parse(self, text: str) -> ParseResult:
        """Parse text and extract tool calls.

        Args:
            text: Raw LLM output text to parse

        Returns:
            ParseResult containing extracted tool calls and metadata
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
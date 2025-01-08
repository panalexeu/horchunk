from typing import Any
from abc import ABC, abstractmethod

from chromadb import EmbeddingFunction

from .chunk import Chunk


class BaseChunker(ABC):
    """
    Abstract base class for chunkers.

    To create a custom chunker, implement the ``__call__`` method.
    Optionally, override the ``tune`` method if tuning functionality is required.
    """

    def __init__(
            self,
            ef: EmbeddingFunction
    ):
        self.ef = ef

    @abstractmethod
    def __call__(self, splits: list[str]) -> list[Chunk]:
        pass

    @abstractmethod
    def tune(self, *args: Any, **kwargs: Any) -> Any:
        """
        Abstract method for tuning the chunker.

        Subclasses should implement this to define custom tuning logic.
        The arguments and return type are flexible to support varied use cases.
        """
        pass

    @abstractmethod
    def eval(self, *args: Any, **kwargs: Any) -> Any:
        """
        Abstract method for evaluating the chunker.

        Subclasses should implement this to define custom evaluation logic.
        The arguments and return type are flexible to accommodate various evaluation scenarios.
        """
        pass

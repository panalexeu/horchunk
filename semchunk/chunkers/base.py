from typing import Any
from abc import ABC, abstractmethod

from chromadb import EmbeddingFunction

from .chunk import Chunk
from .dist import DistanceStrategy


class BaseChunker(ABC):
    """
    Abstract base class for chunkers.

    To create a custom chunker, implement the ``__call__`` method.
    """

    def __init__(
            self,
            ef: EmbeddingFunction,
            df: DistanceStrategy
    ):
        self.ef = ef
        self.df = df

    @abstractmethod
    def __call__(self, splits: list[str]) -> list[Chunk]:
        pass

    @abstractmethod
    def eval(self, *args: Any, **kwargs: Any) -> Any:
        """
        Abstract method for evaluating the chunker.

        Subclasses should implement this to define custom evaluation logic.
        The arguments and return type are flexible to accommodate various evaluation scenarios.
        """
        pass


class BaseTuner(ABC):
    """
    Abstract base class for chunker tuners.

    Subclasses should implement the ``__call__`` method to define custom tuning logic.
    The arguments and return type of the ``__call__`` method are flexible to support varied use cases.

    It is recommended to implement ``BaseTuner`` in the same file where the relevant ``BaseChunker`` is defined.
    """

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Any:
        pass

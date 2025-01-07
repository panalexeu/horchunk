from abc import ABC, abstractmethod

from chromadb import EmbeddingFunction

from .chunk import Chunk


class BaseChunker(ABC):
    """
    Abstract base class for chunkers.

    To create a custom chunker, implement the ``__call__`` method.
    """

    def __init__(
            self,
            ef: EmbeddingFunction
    ):
        self.ef = ef

    @abstractmethod
    def __call__(self, splits: list[str]) -> list[Chunk]:
        pass

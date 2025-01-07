from typing import Callable
from abc import ABC, abstractmethod

from chromadb.api.types import EmbeddingFunction
from ..splitters.base import BaseSplitter


class BaseChunker(ABC):
    """
    Abstract base class for chunkers.

    To create a custom chunker, implement the ``__call__`` method.
    """

    def __init__(
            self,
            ef: EmbeddingFunction
    ):
        self._ef = ef

    @abstractmethod
    def __call__(self, splits: list[str]) -> list[str]:
        pass

    def __or__(self, other: Callable | BaseSplitter) -> list[str]:
        return self.__call__(other.__call__())

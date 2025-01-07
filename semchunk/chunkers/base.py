from abc import ABC, abstractmethod

from chromadb.api.types import EmbeddingFunction


class BaseChunker(ABC):
    def __init__(
            self,
            ef: EmbeddingFunction
    ):
        self._ef = ef

    @abstractmethod
    def __call__(self, text: str) -> list[str]:
        pass

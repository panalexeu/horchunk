from abc import ABC, abstractmethod
from sentence_transformers import SimilarityFunction

from numpy import ndarray


class DistanceStrategy(ABC):

    @abstractmethod
    def calc(self, embed1: ndarray, embed2: ndarray) -> float:
        pass


class CosineDistance(DistanceStrategy):

    def calc(self, embed1: ndarray, embed2: ndarray) -> float:
        cosine = SimilarityFunction.to_similarity_fn('cosine')
        return float(cosine(embed1, embed2).numpy()[0][0])

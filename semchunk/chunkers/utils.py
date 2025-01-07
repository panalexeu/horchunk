from numpy import ndarray, float32
from sentence_transformers import SimilarityFunction


def cosine_dist(
        a: ndarray,
        b: ndarray
) -> float:
    cosine = SimilarityFunction.to_similarity_fn('cosine')
    return cosine(a, b).numpy()[0][0]

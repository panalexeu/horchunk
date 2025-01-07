from chromadb import EmbeddingFunction

from .chunk import Chunk
from .base import BaseChunker
from .utils import cosine_dist


class WindowChunker(BaseChunker):

    def __init__(
            self,
            ef: EmbeddingFunction,
            thresh: float = 0.9
    ):
        super().__init__(ef)
        self.thresh = thresh

    def __call__(self, splits: list[str]) -> list[Chunk]:
        prev = ''
        init = splits[0]
        chunks = []

        for sentence in splits:
            res = prev + ' ' + sentence
            dist = cosine_dist(
                self.ef([init])[0],
                self.ef([res])[0]
            )

            if dist < self.thresh:
                print('formed chunk: ', prev)
                print('brk: ', sentence)
                print('dist: ', dist)
                print('=' * 50)

                chunks.append(prev)
                prev = sentence
                init = sentence
            else:
                prev = res

        if prev not in chunks:
            chunks.append(prev)

        # map chunks to list[Chunk]
        return [Chunk(splits=[chunk]) for chunk in chunks]

import math

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
        prev = splits[0]
        init = splits[0]
        chunks = []

        for sentence in splits:
            if init == sentence:
                continue

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

        # include last chunk
        if prev not in chunks:
            chunks.append(prev)

        # map chunks to list[Chunk]
        return [Chunk(splits=[chunk]) for chunk in chunks]

    def tune(self, splits: list[str], depth: int = 6) -> float:
        """
        Tunes window chunker threshold, by calculating the distance for a bunch of chunks with the ``depth`` splits size.

        Chunks are then sorted with the relevant distances. With the binary search algorithm you decide wich thresh
        should be included, by analyzing created chunks and distance within it.
        """

        if len(splits) < depth:
            raise ValueError(
                "The length of 'splits' is smaller than the defined depth. "
                "Increase the size of the 'splits' list or decrease the 'depth' parameter."
            )

        # consecutively forming key: [distance], value: [chunk] dictionary
        chunks = dict()
        for i in range(0, len(splits), depth):
            init = splits[i]
            res = ' '.join(splits[i:i + depth])

            dist = cosine_dist(
                self.ef([init])[0],
                self.ef([res])[0]
            )

            chunks[dist] = res

        # binary search, with the user evaluation
        sorted_keys = sorted(chunks.keys())

        print(f'{len(sorted_keys)} chunks formed')
        print(f'Values range: [{sorted_keys[0]} ... {sorted_keys[-1]}]')
        time_complex = int(math.log(len(sorted_keys), 2))
        print(f'Steps to tune: {time_complex}')
        print('-' * 16)

        low = 0
        high = len(sorted_keys) - 1
        dist = None
        while low < high:
            mid = (low + high) // 2
            dist = sorted_keys[mid]
            chunk = chunks[dist]

            # reading user evaluation
            print(f'dist: {dist}')
            print(f'chunk: {chunk}')
            input_ = str(input("Type 'h' to raise thresh, or 'l' - to lower it: "))
            print('=' * 32)

            # based on input_ decide what to do next
            match input_:
                case 'h':
                    low = mid + 1
                case 'l':
                    high = mid - 1
                case _:
                    raise IOError('Irrelevant symbol')

        # tuning complete
        print(f'Tuning ended, thresh value: {dist}')

        return dist

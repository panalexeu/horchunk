import math
from typing import Any

from chromadb import EmbeddingFunction
from rich import print

from .chunk import Chunk
from .base import BaseChunker
from .utils import cosine_dist


class WindowChunker(BaseChunker):

    def __init__(
            self,
            ef: EmbeddingFunction,
            thresh: float = 0.72,
            max_chunk_size: int = 6
    ):
        super().__init__(ef)
        self.thresh = thresh
        self.max_chunk_size = max_chunk_size

    def __call__(self, splits: list[str]) -> list[Chunk]:
        cur_chunk = Chunk([])
        init = splits[0]
        chunks: list[Chunk] = []

        for i, sentence in enumerate(splits):
            res = f'{cur_chunk.join()} {sentence}'

            dist = cosine_dist(
                self.ef([init])[0],
                self.ef([res])[0]
            )

            if (dist < self.thresh) or (cur_chunk.size >= self.max_chunk_size):
                print('formed chunk: ', cur_chunk)
                print('brk: ', sentence)
                print('dist: ', dist)
                print('=' * 50)

                chunks.append(cur_chunk)
                cur_chunk = Chunk([sentence])
                init = sentence
            else:
                cur_chunk.add(sentence)

        # include last chunk
        if cur_chunk not in chunks:
            chunks.append(cur_chunk)

        return chunks

    def tune(self, splits: list[str], depth: int = 3) -> float:
        """
        Tunes the window chunker threshold by calculating the distance for a set of chunks with the given ``depth`` split size.

        The chunks are sorted based on the calculated distances. Using a binary search algorithm, the threshold is determined
        by analyzing the created chunks and the distances within them.
        """

        if len(splits) < depth:
            raise ValueError(
                "The length of 'splits' is smaller than the defined depth. "
                "Increase the size of the 'splits' list or decrease the 'depth' parameter."
            )

        # consecutively forming key: [distance], value: [chunk] dictionary
        chunks = dict()
        for i in range(len(splits)):
            init = splits[i]
            chunk = Chunk(splits[i:i + depth])
            res = chunk.join()

            dist = cosine_dist(
                self.ef([init])[0],
                self.ef([res])[0]
            )

            chunks[dist] = chunk

        # binary search, with the user evaluation
        sorted_keys = sorted(chunks.keys())

        # print init tuning params
        print(f'{len(sorted_keys)} chunks formed')
        print(f'Values range: [{sorted_keys[0]} ... {sorted_keys[-1]}]')
        o_n = int(math.log(len(sorted_keys), 2))
        print(f'Steps to tune: {o_n}')
        print('-' * 32)

        # binary search
        low = 0
        high = len(sorted_keys) - 1
        dist = None
        while low <= high:
            mid = (low + high) // 2
            dist = sorted_keys[mid]
            chunk = chunks[dist]

            # printing current chunk info
            print(f'dist: {dist}')
            print(f'chunk: [white on green]{chunk.splits[0]}[/white on green]'
                  f'[white on cyan]{' '.join(chunk.splits[1:])}[white on cyan/]')

            # no need to continue
            if low == high:
                break

            # reading user evaluation
            while True:
                input_ = str(input("Type 'k' to raise thresh, or 'j' - to lower it, then press 'Enter': "))
                if input_ in ['k', 'j']:
                    break
                else:
                    print("Invalid input, please type 'k' or 'j'")
            print('=' * 64)

            # based on input_ decide what to do next
            match input_:
                # raising
                case 'k':
                    low = mid + 1
                # lowering
                case 'j':
                    high = mid - 1

        # tuning complete
        trunc_dist = f'{dist:.3f}'[:4]  # truncating the distance without rounding it
        print(f'Tuning ended, thresh value: {dist} = {trunc_dist}')

        return dist

    def eval(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

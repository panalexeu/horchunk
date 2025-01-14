import math
from enum import Enum
from typing import Any, Callable
from overrides import override

from chromadb import EmbeddingFunction
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from rich import print
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from .base import BaseChunker, BaseTuner
from .chunk import Chunk
from .dist import DistanceStrategy, CosineDistance


class WindowChunker(BaseChunker):

    def __init__(
            self,
            ef: EmbeddingFunction,
            df: DistanceStrategy = CosineDistance(),
            thresh: float = 0.72,
            max_chunk_size: int = 6
    ):
        super().__init__(ef, df)
        self.thresh = thresh
        self.max_chunk_size = max_chunk_size

    def __call__(self, splits: list[str]) -> list[Chunk]:
        cur_chunk = Chunk([])
        init = splits[0]
        chunks: list[Chunk] = []

        for i, sentence in enumerate(splits):
            res = f'{cur_chunk.join()} {sentence}'

            embed_init = self.ef([init])[0]
            embed_res = self.ef([res])[0]

            dist = self.df.calc(embed_init, embed_res)

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

    def eval(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


class WindowTuner(BaseTuner):
    """
    Tunes the window chunker threshold by calculating the distance for a set of chunks with the given ``depth`` split size.

    The chunks are sorted based on the calculated distances. Using a binary search algorithm, the threshold is determined
    by analyzing the created chunks and the distances within them.
    """

    def __init__(
            self,
            ef: EmbeddingFunction,
            df: DistanceStrategy = CosineDistance(),
    ):
        super().__init__(ef, df)

    class SearchCmd(Enum):
        HIGH = 0
        LOW = 1

    def _bin_search(
            self,
            distances: list[float],
            chunks: dict[float, Chunk],
            callback: Callable[[int, list[float], dict[float, Chunk]], SearchCmd]
    ) -> int:
        """
        Performs a binary search on a sorted ``distances`` list using a callback.

        :param distances: a sorted list of distances to search.
        :param chunks:  a dictionary mapping distances to ``Chunk`` objects.
        :param callback: function that returns ``SearchCmd`` enum to direct the binary search.
        :return: the index ``mid`` of the result in ``distances``.
        """
        low = 0
        high = len(distances) - 1
        mid = None
        while low <= high:
            mid = (low + high) // 2

            cmd = callback(mid, distances, chunks)
            match cmd:
                case self.SearchCmd.HIGH:
                    low = mid + 1
                case self.SearchCmd.LOW:
                    high = mid - 1

        return mid

    def __call__(self, splits: list[str], depth: int = 3) -> float:
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

            dist = self.df.calc(
                self.ef([init])[0],
                self.ef([res])[0]
            )

            chunks[dist] = chunk

        # binary search
        sorted_keys = sorted(chunks.keys())
        # print init tuning binary search params
        print(f'{len(sorted_keys)} chunks formed')
        print(f'Values range: [{sorted_keys[0]} ... {sorted_keys[-1]}]')
        o_n = int(math.log(len(sorted_keys), 2))
        print(f'Steps to tune: {o_n}')
        print('-' * 32)

        res_indx = self._bin_search(
            distances=sorted_keys,
            chunks=chunks,
            callback=self._input
        )
        res = sorted_keys[res_indx]

        # search complete
        trunc_dist = f'{res:.3f}'[:4]  # truncating the distance without rounding it
        print(f'Tuning ended, thresh value: {res} = {trunc_dist}')

        return res

    def _input(self, mid: int, distances: list[float], chunks: dict[float, Chunk]) -> SearchCmd:
        # retrieving distance and relevant chunk value
        dist = distances[mid]
        chunk = chunks[dist]

        # printing current chunk info
        print(f'dist: {dist}')
        print(f'chunk: [white on green]{chunk.splits[0]}[/white on green]'
              f'[white on cyan]{' '.join(chunk.splits[1:])}[white on cyan/]')

        # reading the user input
        while True:
            input_ = str(input("Type 'k' to raise thresh, or 'j' - to lower it, then press 'Enter': "))
            match input_:
                case 'k':
                    return self.SearchCmd.HIGH
                case 'j':
                    return self.SearchCmd.LOW
                case _:
                    print("Invalid input, please type 'k' or 'j'")
            print('=' * 64)


class LLMWindowTuner(WindowTuner):
    class Response(BaseModel):
        answer: bool = Field(description='Semantic meaning text classification result.')

    parser_ = PydanticOutputParser(pydantic_object=Response)
    prompt_text = """You are a helpful AI assistant, that classifies the provided text based on the semantic meaning similiarity. 
    If the text is uniform, and shares the same semantic meaning return "true". Otherwise, return "false".
    '''
    {text}
    '''
    {format_instructions}
    """
    prompt = PromptTemplate(
        template=prompt_text,
        input_variables=['text'],
        partial_variables={'format_instructions': parser_.get_format_instructions()}
    )

    def __init__(
            self,
            ef: EmbeddingFunction,
            model: BaseChatModel,
            df: DistanceStrategy = CosineDistance(),
            llm_calls: int = 3
    ):
        super().__init__(ef, df)
        self.llm_calls = llm_calls
        self.model = model

    @override
    def _input(self, mid: int, distances: list[float], chunks: dict[float, Chunk]) -> WindowTuner.SearchCmd:
        # retrieving distance and relevant chunk value
        dist = distances[mid]
        chunk = chunks[dist]

        # printing current chunk info
        print(f'dist: {dist}')
        print(f'chunk: [white on green]{chunk.splits[0]}[/white on green]'
              f'[white on cyan]{' '.join(chunk.splits[1:])}[white on cyan/]')

        # forming requests to llm
        chain = self.prompt | self.model | self.parser_
        responses: list[bool] = []
        for _ in range(self.llm_calls):
            response = chain.invoke({'text': chunk.join()})
            responses.append(response)

        # true => lower, false => raise
        lower_votes = sum(responses)
        raise_votes = self.llm_calls - lower_votes
        action = self.SearchCmd.LOW if lower_votes >= raise_votes else self.SearchCmd.HIGH
        print(f'{lower_votes} llms vote for lowering the thresh')
        print(f'{raise_votes} llms vote for raising')
        print(f'result: {'lower' if action.value else 'raise'}')  # SearchCmd.LOW = 1, SearchCmd.HIGH = 0
        input('Press ENTER to continue: ')

        return action

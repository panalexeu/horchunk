from chromadb.utils import embedding_functions
from rich import print

from semchunk.chunkers import WindowChunker


def test_win_chunker():
    ef = embedding_functions.DefaultEmbeddingFunction()  # all-MiniLM-L6-v2
    chunker = WindowChunker(ef)
    splits = ['Hey!', 'How are you?']
    res = chunker(splits)
    print(res)

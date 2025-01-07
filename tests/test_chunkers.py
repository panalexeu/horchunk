from chromadb.utils import embedding_functions
from rich import print

from semchunk.chunkers import WindowChunker


def test_win_chunker():
    ef = embedding_functions.DefaultEmbeddingFunction()  # all-MiniLM-L6-v2
    chunker = WindowChunker(ef)
    splits = ['Hey!', 'Andromeda']

    chunks = chunker(splits)
    chunks = [chunk.splits[0] for chunk in chunks]

    assert sorted(chunks) == sorted(splits)


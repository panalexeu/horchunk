from chromadb.utils import embedding_functions

from semchunk.chunkers import WindowChunker


def test_win_chunker():
    ef = embedding_functions.DefaultEmbeddingFunction()
    chunker = WindowChunker(ef)
    splits = ['Hey!', 'How are you?']
    chunker(splits)

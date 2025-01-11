import pytest
from chromadb.utils import embedding_functions
from rich import print

from semchunk.chunkers import WindowChunker


@pytest.fixture
def win_chunker():
    ef = embedding_functions.DefaultEmbeddingFunction()  # all-MiniLM-L6-v2
    return WindowChunker(ef)


def test_successfully_chunks(win_chunker):
    splits = ['Hey!', 'Andromeda']

    chunks = win_chunker(splits)
    chunks = [chunk.splits[0] for chunk in chunks]

    assert sorted(chunks) == sorted(splits)


def test_tune_is_not_performed_on_splits_list_smaller_then_depth(win_chunker):
    with pytest.raises(ValueError):
        win_chunker.tune(splits=['hey', 'hey'], depth=3)

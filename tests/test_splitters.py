from tiktoken import get_encoding

from horchunk.splitters import (
    ParagraphSplitter,
    SentenceSplitter,
    TokenSplitter
)


def test_paragraph_splitter():
    text = 'hey.\nhey?\n\nhey!\n\n\n'
    splitter = ParagraphSplitter(text)
    assert ['hey.', 'hey?', 'hey!'] == splitter()


def test_sentence_splitter():
    text = 'hey.  hey?   hey!'
    splitter = SentenceSplitter(text)
    assert ['hey.', 'hey?', 'hey!'] == splitter()


def test_token_splitter():
    text = 'superstar\n\nsubterrain'
    splitter = TokenSplitter(
        text,
        chunk_size=1,
        encoding=get_encoding('cl100k_base')
    )

    assert ['super', 'star', 'sub', 'terrain'] == splitter()

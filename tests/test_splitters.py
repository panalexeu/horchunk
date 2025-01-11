from tiktoken import get_encoding

from semchunk.splitters import SentenceSplitter, TokenSplitter



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
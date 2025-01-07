from semchunk.splitters import SentenceSplitter


def test_sentence_splitter():
    text = 'hey.  hey?   hey!'
    splitter = SentenceSplitter(text)
    assert ['hey.', 'hey?', 'hey!'] == splitter()


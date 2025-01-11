from re import split
from abc import ABC, abstractmethod

from tiktoken import Encoding, get_encoding


class BaseSplitter(ABC):

    def __init__(self, text: str):
        self.text = text

    @abstractmethod
    def __call__(self) -> list[str]:
        pass


class SentenceSplitter(BaseSplitter):
    """Splits the text into paragraphs first, then into sentences using punctuation marks ``.!?``."""

    def __call__(self) -> list[str]:
        paragraphs = split(r'\n+', self.text)

        sentences = []
        for paragraph in paragraphs:
            split_ = split(r'(?<=[.!?])\s+', paragraph)
            sentences.extend([sentence for sentence in split_ if sentence.strip()])

        return sentences


class TokenSplitter(BaseSplitter):
    """
    Splits the text into paragraphs first. If a resulting paragraph exceeds the specified number of tokens,
    it is further split into smaller chunks, each of ``chunk_size`` tokens.
    """

    def __init__(
            self,
            text: str,
            chunk_size: int = 32,
            encoding: Encoding = get_encoding('cl100k_base')
    ):
        """
        :param text: text to split.
        :param chunk_size: chunk size in tokens.
        :param encoding: Byte-Pair encoding to be used. Accepts ``Encoding`` object from the ``tiktoken`` package.
        By default, ``cl100k_base`` is used.
        """
        super().__init__(text)
        self.chunk_size = chunk_size
        self.encoding = encoding

    def __call__(self) -> list[str]:
        paragraphs = split(r'\n+', self.text)

        splits_ = []
        for paragraph in paragraphs:
            tokens = self.encoding.encode(paragraph)
            if len(tokens) > self.chunk_size:
                for i in range(0, len(tokens), self.chunk_size):
                    chunk = tokens[i:i + self.chunk_size]
                    decoded_chunk = self.encoding.decode(chunk)
                    splits_.append(decoded_chunk)

        return splits_

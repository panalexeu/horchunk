import re

from .base import BaseSplitter


class SentenceSplitter(BaseSplitter):
    """Splits the text into paragraphs first, then into sentences using punctuation marks ``.!?``."""

    def __call__(self) -> list[str]:
        paragraphs = re.split(r'\n+', self.text)

        sentences = []
        for paragraph in paragraphs:
            split_ = re.split(r'(?<=[.!?])\s+', paragraph)
            sentences.extend(split_)

        return sentences

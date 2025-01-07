import re

from .base import BaseSplitter


class SentenceSplitter(BaseSplitter):
    def __call__(self) -> list[str]:
        pattern = r'(?<=[.!?])\s+'
        return re.split(pattern, self.text)

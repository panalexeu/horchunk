from abc import ABC, abstractmethod


class BaseSplitter(ABC):

    def __init__(self, text: str):
        self.text = text

    @abstractmethod
    def __call__(self) -> list[str]:
        pass

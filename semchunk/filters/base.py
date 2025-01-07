from abc import ABC, abstractmethod


class BaseFilter(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def __call__(self):
        pass

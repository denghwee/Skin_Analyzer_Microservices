from abc import ABC, abstractmethod

class BaseService(ABC):
    """
    Interface gốc cho mọi classification service của hệ thống.
    """

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def classify(self):
        pass
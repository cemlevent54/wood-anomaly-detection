from abc import ABC, abstractmethod

class ModelInterface(ABC):
    @abstractmethod
    def test_model_with_photo(self):
        pass
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple
from numbers import Number
import unicodedata


def remove_diacritics(text):
    normalized_text = unicodedata.normalize('NFD', text)
    stripped_text = ''.join(char for char in normalized_text if unicodedata.category(char) != 'Mn')
    return stripped_text


class Model(ABC):
    @abstractmethod
    def save(self, path: str) -> None:
        pass
    @abstractmethod
    def load(self, path: str) -> bool:
        pass
    @abstractmethod
    def train(self, training_length: float) -> Tuple[float, Dict[str, float]]:
        pass

class ModelGenerator(ABC):
    @abstractmethod
    def get_model(self, params: Dict[str, Any]) -> Model:
        pass

class ModelTrainer(ABC):
    @abstractmethod
    def train(self, training_length: float) -> Tuple[float, Dict[str, float]]:
        pass

class TrainerGenerator(ABC):
    @abstractmethod
    def get_trainer(self, params: Dict[str, Any], filename: str) -> ModelTrainer:
        pass

class ParameterSampler(ABC):
    @abstractmethod
    def sample_parameters(self) -> Dict[str, Any]:
        pass


class LetterEncoder:
    def __init__(self, alphabet_dictionary):
        self.alphabet_dictionary = alphabet_dictionary

    def __len__(self):
        return len(self.alphabet_dictionary) + 1

    def encode_letter(self, letter):
        if letter in self.alphabet_dictionary:
            return self.alphabet_dictionary[letter]
        else:
            return self.alphabet_dictionary['other']



import torch
from torch.nn.utils.rnn import pad_sequence
from abc import ABC, abstractmethod
from typing import Dict, List


class Tokenizer(ABC):
    """
    Abstract base class for tokenizers

    """
    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def encode(self, input_data: str) -> torch.Tensor:
        pass

    @abstractmethod
    def batch_encode(self, input_data: List[str], padding: str = 'longest') -> torch.Tensor:
        pass

    @abstractmethod
    def batch_encode_plus(self, input_data: List[str], padding: str = 'longest') -> Dict:
        pass


class CharacterTokenizer(Tokenizer):
    def __init__(self):
        self.special_tokens = {
            '<NULL>': 0,
            '<EOS>': 1,
            '<UNK>': 2
        }

    def __len__(self):
        return 256 + len(self.special_tokens)

    def add_token(self, token: str):
        self.special_tokens[token] = len(self.special_tokens)

    def get_special_tokens(self) -> Dict[str, int]:
        return self.special_tokens

    def encode(self, input_data: str) -> torch.Tensor:
        encoded = [self.special_tokens['<EOS>']] + [_id + len(self.special_tokens) for _id in list(input_data.encode('utf-8'))] + [self.special_tokens['<EOS>']]
        return torch.tensor(encoded)

    def batch_encode(self, input_data: List[str], padding: str = 'longest') -> torch.Tensor:
        sequences = [self.encode(seq) for seq in input_data]
        encoded = pad_sequence(sequences, batch_first=True, padding_value=self.special_tokens['<NULL>'])
        return encoded

    def batch_encode_plus(self, input_data: List[str], padding: str = 'longest') -> Dict:
        sequences = [self.encode(seq) for seq in input_data]
        lengths = torch.tensor([len(seq) for seq in sequences])
        encoded = pad_sequence(sequences, batch_first=True, padding_value=self.special_tokens['<NULL>'])
        return {'input_ids': encoded, 'lengths': lengths}
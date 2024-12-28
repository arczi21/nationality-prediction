import re
import pandas as pd
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader


LETTERS = " -'abcdefghijklmnopqrstuvwxyzßàáâãäåæçèéêëìíîïñòóôõöøùúûüýÿăąćčďęěĺľłńňőœŕřśšťůűźżžșțαβγδεζηθικλμνξοπρστυφχψωабвгдежзийклмнопрстуфхцчшщъыьэюяёєіїўґ"


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
    def __init__(self, letters):
        alphabet = ''.join(list(sorted(set(letters))))
        self.alphabet_dictionary = {}
        self.add_character('')
        for c in sorted(alphabet):
            self.add_character(c)
        self.add_character('<OTHER>')
        self.add_character('<EOS>')
        self.add_character('<UNK>')

    def __len__(self):
        return len(self.alphabet_dictionary)

    def add_character(self, c):
        self.alphabet_dictionary[c] = len(self.alphabet_dictionary)

    def encode_letter(self, letter):
        if letter in self.alphabet_dictionary:
            return self.alphabet_dictionary[letter]
        else:
            return self.alphabet_dictionary['<OTHER>']


class NamesDataset(Dataset):
    def __init__(self, names, labels, nat_labels, letter_encoder):
        self.names = names
        self.labels = labels
        self.label_map = {label: i for i, label in enumerate(nat_labels)}
        self.letter_encoder = letter_encoder

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        label = self.label_map[self.labels[idx]]
        encoded_word = [self.letter_encoder.encode_letter(c) for c in name.lower()]
        return torch.tensor(encoded_word, dtype=torch.long), torch.tensor(len(encoded_word), dtype=torch.long), torch.tensor(label, dtype=torch.long)


def preprocessing(data):

    # Removing brackets in name, for example: "Jan Kowalski (blacksmith)" -> "Jan Kowalski"
    data.name = data.name.apply(lambda x: re.sub(r' \(.*?\)', '', x))

    # Removing whitespaces before and after the name
    data.name = data.name.apply(lambda x: x.strip())

    data = data.reset_index(drop=True)

    return data


def load_data(train_data_path='data/csv/nationalities.csv', test_data_path='data/csv/nationalities_test.csv'):
    train_df = pd.read_csv(train_data_path)
    test_df = pd.read_csv(test_data_path)

    train_df = preprocessing(train_df)
    test_df = preprocessing(test_df)

    train_df = train_df.drop(train_df[train_df['name'].isin(test_df['name'])].index).reset_index(drop=True)

    return train_df, test_df


def get_labels(dataframe):
    return list(dataframe['nationality'].unique())


def prepare_data(dataframe, nat_labels, letter_encoder, batch_size, valid_size=10000):
    dataset = NamesDataset(dataframe.name, dataframe.nationality, nat_labels, letter_encoder)
    train_size = len(dataframe) - valid_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size], generator=generator)

    def collate_fn(batch):
        words, lengths, labels = zip(*batch)
        padded_words = pad_sequence(words, batch_first=True, padding_value=0)
        return padded_words, torch.tensor(lengths), torch.tensor(labels)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    return train_dataloader, valid_dataloader

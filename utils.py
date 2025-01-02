import re
import pandas as pd
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import torch
from torch.utils.data import Dataset, DataLoader




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


class NamesDataset(Dataset):
    def __init__(self, names, labels, nat_labels):
        self.names = names
        self.labels = labels
        self.label_map = {label: i for i, label in enumerate(nat_labels)}

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        label = self.label_map[self.labels[idx]]
        return name, label


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


def prepare_data(dataframe, nat_labels, batch_size=50, valid_size=20000):
    dataset = NamesDataset(dataframe['name'], dataframe['nationality'], nat_labels)
    train_size = len(dataframe) - valid_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size], generator=generator)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader, valid_dataloader

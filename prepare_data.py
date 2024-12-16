import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from collections import Counter

from preprocessing import preprocessing
from utils import LetterEncoder


class NamesDataset(Dataset):
    def __init__(self, names, labels, label_map, letter_encoder):
        self.names = names
        self.labels = labels
        self.label_map = label_map
        self.letter_encoder = letter_encoder

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        label = self.label_map[self.labels[idx]]
        encoded_word = [self.letter_encoder.encode_letter(c) for c in name.lower()]
        return torch.tensor(encoded_word, dtype=torch.long), torch.tensor(len(encoded_word), dtype=torch.long), torch.tensor(label, dtype=torch.long)


def collate_fn(batch):
    words, lengths, labels = zip(*batch)
    padded_words = pad_sequence(words, batch_first=True, padding_value=0)
    return padded_words, torch.tensor(lengths), torch.tensor(labels)


def prepare_and_load_data(filename):
    data = pd.read_csv(filename)
    data = preprocessing(data)

    letter_counter = Counter(data.name.str.cat().lower())
    occurrence_bound = 50
    all_characters = set(data.name.str.cat().lower())

    alphabet_dictionary = {}
    c = 1
    for s in sorted(all_characters):
        if letter_counter[s] > occurrence_bound:
            alphabet_dictionary[s] = c
            c += 1
    alphabet_dictionary['other'] = len(alphabet_dictionary) + 1

    letter_encoder = LetterEncoder(alphabet_dictionary)

    nat_labels = list(data.nationality.unique())
    label_map = {label: i for i, label in enumerate(nat_labels)}
    dataset = NamesDataset(data.name, data.nationality, label_map, letter_encoder)

    valid_len = 20000
    test_len = 20000
    train_len = len(dataset) - valid_len - test_len

    generator = torch.Generator().manual_seed(42)
    train_set, valid_set, test_set = torch.utils.data.random_split(dataset, [train_len, valid_len, test_len],
                                                                        generator=generator)

    train_dataloader = DataLoader(train_set, batch_size=50, shuffle=True, collate_fn=collate_fn)
    valid_dataloader = DataLoader(valid_set, batch_size=50, shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_set, batch_size=50, shuffle=True, collate_fn=collate_fn)

    return train_dataloader, valid_dataloader, test_dataloader, letter_encoder, nat_labels



import os
import wandb
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle

from torch.utils.data import DataLoader
from typing import Any, Dict
from numbers import Number

from utils import ModelTrainer
from prepare_data import prepare_and_load_data
from models import GRU, LSTM, RNN


device = torch.device('cuda')


class EphemeralModelTrainer(ModelTrainer):
    def __init__(self, Class: Any, params: Dict[str, Any], train_dataloader: DataLoader, valid_dataloader: DataLoader,
                 log_every: int, filename='test_model', log_wandb=False, epoch_save=False):

        self.Class = Class
        self.lr = params['lr']
        self.network_params = params.copy()
        del self.network_params['lr']

        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.log_every = log_every
        self.filename = filename
        self.log_wandb = log_wandb
        self.epoch_save = epoch_save

        self.criterion = nn.CrossEntropyLoss()
        self.current_training_length = 0
        self.iteration = 0

        self.model = None
        self.optimizer = None

        if self.log_wandb:
            wandb.init(
                project="nationality-prediction",
                name=filename,
                config=params)

    def load(self):
        self.model = self.Class(**self.network_params).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        if os.path.exists(f"checkpoints/{self.filename}/model.pth"):
            checkpoint = torch.load(f"checkpoints/{self.filename}/model.pth")
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            return True
        return False

    def save(self, path, clear_memory=True):
        if os.path.dirname(path) != '':
            os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'params': self.network_params,
        }, self.filename)

        if clear_memory:
            self.model = None
            self.optimizer = None

    def test(self, dataloader: DataLoader):
        losses = []
        accuracies = []
        for sequences, lengths, labels in dataloader:
            sequences = sequences.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                out = self.model(sequences, lengths)
                test_loss = self.criterion(out, labels)
                argmax = out.argmax(dim=1)

            losses.append(test_loss.item())
            accuracies.extend((argmax == labels).float().cpu().numpy())

        return np.mean(losses), {'accuracy': np.mean(accuracies)}

    def train(self, training_length: float):

        self.load()

        it = iter(self.train_dataloader)

        while self.current_training_length < training_length:

            try:
                sequences, lengths, labels = next(it)
            except StopIteration:
                if self.epoch_save:
                    self.save(path=f"checkpoints/{self.filename}/model.pth", clear_memory=False)
                    self.save(path=f"checkpoints/{self.filename}/ckpt/model_step{self.iteration}.pth", clear_memory=False)
                it = iter(self.train_dataloader)
                sequences, lengths, labels = next(it)

            sequences = sequences.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)

            self.model.train()

            outputs = self.model(sequences, lengths)
            loss = self.criterion(outputs, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.log_wandb:
                wandb.log({"train_loss": loss.item()}, step=self.iteration)

            self.current_training_length += sequences.size(0) / len(self.train_dataloader.dataset)
            self.iteration += 1

            if self.iteration % self.log_every == 0:
                self.model.eval()
                test_loss, info = self.test(self.valid_dataloader)
                print(f'{self.current_training_length:.2f}/{training_length:.2f} : test loss: {test_loss}, accuracy: {info["accuracy"]}')
                if self.log_wandb:
                    wandb.log({"test_loss": test_loss, "accuracy": info['accuracy']}, step=self.iteration)


        self.model.eval()
        test_loss, info = self.test(self.valid_dataloader)

        self.save(path=f"checkpoints/{self.filename}/model.pth")

        if self.log_wandb:
            wandb.log({"test_loss": test_loss, "accuracy": info['accuracy']}, step=self.iteration)
            wandb.finish()

        return test_loss


if __name__ == "__main__":
    train_dataloader, valid_dataloader, test_dataloader, letter_encoder, nat_labels = prepare_and_load_data('nationalities.csv')

    params = {
        'input_size': 99,
        'output_size': 24,
        'embedding_dim': 64,
        'hidden_dim': 384,
        'num_layers': 3,
        'final_dropout': 0.3,
        'lr': 0.0001
    }
    num_epochs = 15

    trainer = EphemeralModelTrainer(LSTM, params, train_dataloader, valid_dataloader, 100, filename="gru_0", log_wandb=True)
    trainer.train(num_epochs)

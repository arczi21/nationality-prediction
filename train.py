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

from hyperband import ModelTrainer
from prepare_data import prepare_and_load_data
from models import GRU


device = torch.device('cuda')


class Trainer(ModelTrainer):
    def __init__(self, modelClass: Any, params: Dict[str, Any], train_dataloader: DataLoader, valid_dataloader: DataLoader, log_every: Number,
                filename='model.pth', log_wandb=False):
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.log_every = log_every
        self.params = params
        self.filename = filename
        self.modelClass = modelClass
        self.log_wandb = log_wandb
        self.lr = self.params['lr']
        self.network_params = self.params.copy()
        del self.network_params['lr']
        self.model = None
        self.optimizer = None

        self.criterion = nn.CrossEntropyLoss()
        self.current_training_length = 0
        self.iteration = 0

        if self.log_wandb:
            wandb.init(
                project="nationality-prediction",
                name='gru-model',
                config=self.params)

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

        self.model = self.modelClass(**self.network_params).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        if os.path.exists(self.filename):
            checkpoint = torch.load(self.filename)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        it = iter(self.train_dataloader)

        while self.current_training_length < training_length:

            try:
                sequences, lengths, labels = next(it)
            except StopIteration:
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

        if os.path.dirname(self.filename) != '':
            os.makedirs(os.path.dirname(self.filename), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, self.filename)
        del self.optimizer
        del self.model

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

    from natpred import NationalityPrediction
    nat_pred = NationalityPrediction()

    #trainer = Trainer(GRU, params, train_dataloader, valid_dataloader, 3000, log_wandb=True)
    #trainer.train(num_epochs)

    #with open('model/data.pkl', 'wb') as f:
    #    pickle.dump((letter_encoder, nat_labels), f)

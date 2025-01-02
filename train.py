import os
import wandb
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from torch.utils.data import DataLoader
from typing import Any, Dict

from utils import ModelTrainer, load_data, get_labels, prepare_data
from models import GRU, LSTM, RNN, TransformerEncoderClassifier
from tokenizers import Tokenizer, CharacterTokenizer


class LoadSaveTrainer(ModelTrainer):
    def __init__(self, Class: Any, params: Dict[str, Any], train_dataloader: DataLoader, valid_dataloader: DataLoader,
                 tokenizer: Tokenizer, device: torch.device, filename='test_model', wandb_name='test', log_wandb=False, epoch_save=False):

        self.Class = Class
        self.params = params
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.tokenizer = tokenizer
        self.device = device

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
                name=wandb_name,
                config=params)

    def load(self):
        self.model = self.Class(**self.params).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.params['lr'])
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
            'params': self.params,
        }, path)

        if clear_memory:
            self.model = None
            self.optimizer = None

    def test(self, dataloader: DataLoader):

        y_true = []
        y_pred = []
        losses = []
        criterion = nn.CrossEntropyLoss(reduction='none')

        for sequences, labels in dataloader:
            inp = self.tokenizer.batch_encode_plus(sequences)
            for k, v in inp.items():
                inp[k] = v.to(self.device)
            labels = labels.to(self.device)

            with torch.no_grad():
                out = self.model(**inp)
                argmax = out.argmax(dim=1)

            losses.extend(criterion(out, labels).tolist())
            y_true.extend(list(labels.cpu().numpy()))
            y_pred.extend(list(argmax.cpu().numpy()))

        return {
            'loss': np.mean(losses),
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        }

    def train(self, training_length: float):

        self.load()

        it = iter(self.train_dataloader)

        while self.current_training_length < training_length:

            try:
                sequences, labels = next(it)
            except StopIteration:

                # test model and log
                self.model.eval()
                info = self.test(self.valid_dataloader)
                print(f'{self.current_training_length:.2f}/{training_length:.2f} : ' + ", ".join(
                    f'test_{key}: {value}' for key, value in info.items()))
                if self.log_wandb:
                    wandb.log({f'test_{key}': value for key, value in info.items()}, step=self.iteration)

                if self.epoch_save:
                    self.save(path=f"checkpoints/{self.filename}/model.pth", clear_memory=False)
                    self.save(path=f"checkpoints/{self.filename}/ckpt/model_step{self.iteration}.pth", clear_memory=False)
                it = iter(self.train_dataloader)
                sequences, labels = next(it)

            inp = self.tokenizer.batch_encode_plus(sequences)
            for k, v in inp.items():
                inp[k] = v.to(self.device)
            labels = labels.to(self.device)

            self.model.train()
            outputs = self.model(**inp)
            loss = self.criterion(outputs, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.log_wandb:
                wandb.log({"train_loss": loss.item()}, step=self.iteration)

            self.current_training_length += len(sequences) / len(self.train_dataloader.dataset)
            self.iteration += 1


        self.model.eval()
        info = self.test(self.valid_dataloader)

        self.save(path=f"checkpoints/{self.filename}/model.pth")

        if self.log_wandb:
            wandb.log({f'test_{key}': value for key, value in info.items()}, step=self.iteration)
            wandb.finish()

        return info['f1']


if __name__ == "__main__":

    train_df, test_df = load_data()
    nat_labels = get_labels(train_df)
    train_dataloader, valid_dataloader = prepare_data(train_df, nat_labels, batch_size=50, valid_size=20000)
    tokenizer = CharacterTokenizer()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    params = {
        'input_size': len(tokenizer),
        'output_size': len(nat_labels),
        'embedding_dim': 128,
        'hidden_dim': 256,
        'num_layers': 2,
        'final_dropout': 0.5,
        'lr': 0.002
    }

    num_epochs = 5
    name = 'lstm_test'

    trainer = LoadSaveTrainer(Class=LSTM, params=params, train_dataloader=train_dataloader,
                              valid_dataloader=valid_dataloader, tokenizer=tokenizer, device=device,
                              filename=name, wandb_name=name, log_wandb=True)

    trainer.train(num_epochs)
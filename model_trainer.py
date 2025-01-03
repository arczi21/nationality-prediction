import os
import wandb
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from abc import ABC, abstractmethod
from typing import Tuple
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from torch.utils.data import DataLoader
from typing import Any, Dict
from ntokenizers import Tokenizer


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


class LoadSaveTrainer(ModelTrainer):
    def __init__(self,
                 Class: Any,
                 params: Dict[str, Any],
                 train_dataloader: DataLoader,
                 valid_dataloader: DataLoader,
                 tokenizer: Tokenizer,
                 device: torch.device,
                 filename='test_model',
                 wandb_name='test',
                 log_wandb=False,
                 epoch_save=False):

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
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.params['lr'], weight_decay=self.params['weight_decay'])
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
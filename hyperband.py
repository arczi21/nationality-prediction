import random
import math

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple
from numbers import Number
from dataclasses import dataclass
import numpy as np

from models import GRU
from prepare_data import prepare_and_load_data


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


@dataclass
class StatEntry:
    stat_id: int
    params: Dict[str, Any]
    trainer: ModelTrainer
    loss: float


class HyperbandStatistics:
    def __init__(self):
        self.stats = {}

    def add(self, s: int, params: Dict[str, Any], trainer: ModelTrainer):
        if s not in self.stats:
            self.stats[s] = []
        stat_id = len(self.stats[s])
        self.stats[s].append(StatEntry(stat_id, params, trainer, float('inf')))

    def update_loss(self, s: int, stat_id: int, loss: float):
        not_found = True
        for entry in self.stats[s]:
            if entry.stat_id == stat_id:
                entry.loss = loss
                not_found = False
                break
        if not_found:
            raise Exception(f"No entry with id={stat_id}")

    def get_stats(self, s: int):
        return self.stats[s]

    def top_k(self, s: int, k: int):
        self.stats[s] = sorted(self.stats[s], key=lambda x: x.loss)[:k]
        return self.get_stats(s)
    

class Hyperband:
    def __init__(self, parameter_sampler: ParameterSampler, trainer_generator: TrainerGenerator, R: int = 81, eta: float = 3):
        self.parameter_sampler = parameter_sampler
        self.trainer_generator = trainer_generator
        self.R = R
        self.eta = eta

        self.s_max = math.floor(math.log(R, eta))
        self.B = (self.s_max + 1) * R

        self.s = self.s_max
        self.stats = HyperbandStatistics()

    @property
    def finished(self):
        return self.s < 0

    def calculate_bracket(self, s: int):
        n = math.ceil((self.B / self.R) * (self.eta ** s / (s + 1)))
        r = self.R * self.eta ** (-s)
        for i in range(n):
            params = self.parameter_sampler.sample_parameters()
            filename = f'hyperband/{s}/model_checkpoint_{i}.pth'
            trainer = self.trainer_generator.get_trainer(params, filename)
            self.stats.add(s, params, trainer)

        for i in range(s + 1):
            ni = math.floor(n * self.eta ** (-i))
            ri = r * self.eta ** i

            print(f'##### Starting iteration {i+1}/{s+1} in bracket s={s}: n_i={ni}, r_i={ri}')

            entries = self.stats.top_k(s, ni)
            for j, entry in enumerate(entries):
                loss = entry.trainer.train(ri)
                entry.loss = loss
                print(f'Run {j+1}/{ni} finished with loss={loss}')

        print(f'Best params:')
        entries = self.stats.get_stats(s)
        for entry in entries:
            print(f'Loss: {entry.loss}, Id: {entry.stat_id}, Params: {entry.params}')


    def proceed(self):
        if self.s >= 0:
            self.calculate_bracket(self.s)
            self.s -= 1
        else:
            raise Exception("proceed() was called after Hyperband had already finished.")



if __name__ == "__main__":

    train_dataloader, valid_dataloader, test_dataloader = prepare_and_load_data('nationalities.csv')

    class ClassGenerator(TrainerGenerator):
        def __init__(self, ModelClass: Any, log_every: Number = 100):
            self.Class = ModelClass
            self.log_every = log_every

        def get_trainer(self, params, filename):
            return Trainer(self.Class, params, train_dataloader, valid_dataloader, test_dataloader, self.log_every,
                           filename)


    class Sampler(ParameterSampler):
        def __init__(self):
            self.grid = {
                'input_size': [99],
                'output_size': [24],
                'embedding_dim': [64, 128, 192, 256],
                'hidden_dim': [128, 256, 384, 512, 640, 768],
                'num_layers': [1, 2, 3],
                'final_dropout': [0, 0.3, 0.5],
                'lr': [0.001, 0.0003, 0.0001, 0.00003, 0.00001]
            }

        def sample_parameters(self):
            params = {}
            for key in self.grid.keys():
                idx = np.random.choice(len(self.grid[key]))
                params[key] = self.grid[key][idx]
            return params


    gru_generator = ClassGenerator(GRU, float('inf'))
    sampler = Sampler()

    hyper = Hyperband(sampler, gru_generator, R=10, eta=1.5)
    while not hyper.finished:
        hyper.proceed()
import torch
from torch.utils.data import DataLoader

from model_trainer import ModelTrainer, LoadSaveTrainer
from utils import load_data, get_labels, prepare_data
from models import GRU, LSTM, MultilingualBert
from ntokenizers import CharacterTokenizer, TransformerTokenizer


def get_trainer(model_name: str,
                train_dataloader: DataLoader,
                valid_dataloader: DataLoader,
                n_classes: int = None,
                filename: str = 'test',
                wandb_name: str = 'test',
                **kwargs) -> ModelTrainer:

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model_name == 'lstm':
        tokenizer = CharacterTokenizer()
        params = {
            'input_size': len(tokenizer),
            'output_size': n_classes,
            'embedding_dim': 128,
            'hidden_dim': 512,
            'num_layers': 2,
            'final_dropout': 0.5,
            'lr': 0.002,
            'weight_decay': 0.00003,
        }
        return LoadSaveTrainer(Class=GRU,
                               params=params,
                               train_dataloader=train_dataloader,
                               valid_dataloader=valid_dataloader,
                               tokenizer=tokenizer,
                               device=device,
                               filename=filename,
                               wandb_name=wandb_name)

    elif model_name == 'lstm':
        tokenizer = CharacterTokenizer()
        params = {
            'input_size': len(tokenizer),
            'output_size': n_classes,
            'embedding_dim': 128,
            'hidden_dim': 512,
            'num_layers': 2,
            'final_dropout': 0.5,
            'lr': 0.002,
            'weight_decay': 0.00003,
        }
        return LoadSaveTrainer(Class=LSTM,
                               params=params,
                               train_dataloader=train_dataloader,
                               valid_dataloader=valid_dataloader,
                               tokenizer=tokenizer,
                               device=device,
                               filename=filename,
                               wandb_name=wandb_name)

    elif model_name == 'mbert':
        tokenizer = TransformerTokenizer('bert-base-multilingual-uncased')
        params = {
            'output_size': len(nat_labels),
            'lr': 0.00005,
            'weight_decay': 0,
        }
        return LoadSaveTrainer(Class=MultilingualBert,
                               params=params,
                               train_dataloader=train_dataloader,
                               valid_dataloader=valid_dataloader,
                               tokenizer=tokenizer,
                               device=device,
                               filename=filename,
                               wandb_name=wandb_name)



if __name__ == "__main__":

    train_df, test_df = load_data()
    nat_labels = get_labels(train_df)
    train_dataloader, valid_dataloader = prepare_data(train_df, nat_labels, batch_size=50, valid_size=20000)

    name = 'test_42'
    num_epochs = 10

    trainer = get_trainer(model_name='gru',
                          train_dataloader=train_dataloader,
                          valid_dataloader=valid_dataloader,
                          n_classes=len(nat_labels),
                          filename=name,
                          wandb_name=name)

    trainer.train(num_epochs)
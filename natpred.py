import torch
import torch.nn as nn
import pickle
from utils import LetterEncoder


from models import GRU


class NationalityPrediction:
    def __init__(self, model_path='model', device='cuda'):
        params = {
            'input_size': 99,
            'output_size': 24,
            'embedding_dim': 64,
            'hidden_dim': 384,
            'num_layers': 3,
            'final_dropout': 0.3
        }
        self.device = torch.device(device)
        self.model = GRU(**params).to(self.device)
        checkpoint = torch.load(f'{model_path}/model.pth')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        with open(f'{model_path}/data.pkl', 'rb') as f:
            letter_encoder, nat_labels = pickle.load(f)

        self.nat_labels = nat_labels
        self.letter_encoder = letter_encoder


    def predict(self, x, k=5):
        encoded_word = [self.letter_encoder.encode_letter(char) for char in x.lower()]
        w = torch.tensor([encoded_word], dtype=torch.long).to(self.device)
        out = nn.functional.softmax(self.model(w), dim=1)
        ind = torch.topk(out, k).indices[0].cpu().numpy()
        return {self.nat_labels[idx]: out[0, idx].item() for idx in ind}

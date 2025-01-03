import torch.nn as nn
from transformers import BertModel

class RNN(nn.Module):
    def __init__(self, input_size, output_size, embedding_dim, hidden_dim, num_layers, final_dropout=0):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.embeddings = nn.Embedding(input_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, self.hidden_dim, self.num_layers, batch_first=True)
        self.dropout = nn.Dropout(p=final_dropout)
        self.fc = nn.Linear(self.hidden_dim, output_size)

    def forward(self, x, lengths=None):
        out = self.embeddings(x)
        out, _ = self.rnn(out)
        if lengths is None:
            out = out[:, -1, :]
        else:
            out = out.gather(1,
                             lengths.unsqueeze(-1).unsqueeze(-1).expand(out.size(0), 1, self.hidden_dim) - 1).squeeze(1)
        out = self.dropout(out)
        out = self.fc(out)
        return out


class LSTM(nn.Module):
    def __init__(self, input_size, output_size, embedding_dim, hidden_dim, num_layers, final_dropout=0, **kwargs):
        super(LSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.embeddings = nn.Embedding(input_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, self.num_layers, batch_first=True)
        self.dropout = nn.Dropout(p=final_dropout)
        self.fc = nn.Linear(self.hidden_dim, output_size)

    def forward(self, input_ids, lengths=None, **kwargs):
        out = self.embeddings(input_ids)
        out, _ = self.lstm(out)
        if lengths is None:
            out = out[:, -1, :]
        else:
            out = out.gather(1, lengths.unsqueeze(-1).unsqueeze(-1).expand(out.size(0), 1, self.hidden_dim)-1).squeeze(1)
        out = self.dropout(out)
        out = self.fc(out)
        return out


class GRU(nn.Module):
    def __init__(self, input_size, output_size, embedding_dim, hidden_dim, num_layers, final_dropout=0, **kwargs):
        super(GRU, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.embeddings = nn.Embedding(input_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, self.hidden_dim, self.num_layers, batch_first=True)
        self.dropout = nn.Dropout(p=final_dropout)
        self.fc = nn.Linear(self.hidden_dim, output_size)

    def forward(self, input_ids, lengths=None, **kwargs):
        out = self.embeddings(input_ids)
        out, _ = self.gru(out)
        if lengths is None:
            out = out[:, -1, :]
        else:
            out = out.gather(1, lengths.unsqueeze(-1).unsqueeze(-1).expand(out.size(0), 1, self.hidden_dim)-1).squeeze(1)
        out = self.dropout(out)
        out = self.fc(out)
        return out


class MultilingualBert(nn.Module):
    def __init__(self, output_size, **kwargs):
        super(MultilingualBert, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-multilingual-uncased')
        self.fc = nn.Linear(768, output_size)

    def forward(self, **kwargs):
        outputs = self.bert(**kwargs)
        pooled_output = outputs.pooler_output
        logits = self.fc(pooled_output)
        return logits